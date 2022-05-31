import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from seq2seq import utils


class TransformerEncoderLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.embed_dim = args.encoder_embed_dim
        self.self_attn = MultiHeadAttention(
            self.embed_dim,
            args.encoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=True
        )
        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.dropout = args.attention_dropout
        self.activation_dropout = args.activation_dropout

        self.fc1 = generate_linear(self.embed_dim, args.encoder_ffn_embed_dim)
        self.fc2 = generate_linear(args.encoder_ffn_embed_dim, self.embed_dim)
        self.final_layer_norm = LayerNorm(self.embed_dim)

    def forward(self, state, encoder_padding_mask):
        """Forward pass of a single Transformer Encoder Layer"""
        residual = state.clone()

        '''
        ___QUESTION-6-DESCRIBE-D-START___
        1.  Add tensor shape annotation to EVERY TENSOR below (NOT just the output tensor)
        2.  What is the purpose of encoder_padding_mask? 
        3.  What will the output shape of `state' Tensor be after multi-head attention?
        '''
        #print(state.size())
        state, _ = self.self_attn(query=state, key=state, value=state, key_padding_mask=encoder_padding_mask)
        #print(state.size())
        # state.size = [source sentence length, batch of sentences, embedding features]
        # encoder_padding_mask.size = [batch of sentences, source sentence length]
        #2. The purpose of encoder_padding_mask is to mask sentences that are shorter than the largest sentences in the batch.
        #   This way we have fixed same input size vectors being fed into the transformer encoder on each batch.
        #3. state.size = [source sentence length, batch of sentences, embedding features]
        '''
        ___QUESTION-6-DESCRIBE-D-END___
        '''

        state = F.dropout(state, p=self.dropout, training=self.training)
        state += residual
        state = self.self_attn_layer_norm(state)

        residual = state.clone()
        state = F.relu(self.fc1(state))
        state = F.dropout(state, p=self.activation_dropout, training=self.training)
        state = self.fc2(state)
        state = F.dropout(state, p=self.dropout, training=self.training)
        state += residual
        state = self.final_layer_norm(state)

        return state


class TransformerDecoderLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.embed_dim = args.decoder_embed_dim
        self.dropout = args.attention_dropout
        self.activation_dropout = args.activation_dropout
        self.self_attn_layer_norm = LayerNorm(self.embed_dim)

        self.self_attn = MultiHeadAttention(
            embed_dim=self.embed_dim,
            num_attn_heads=args.decoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=True
        )

        self.encoder_attn = MultiHeadAttention(
            embed_dim=self.embed_dim,
            num_attn_heads=args.decoder_attention_heads,
            kdim=args.encoder_embed_dim,
            vdim=args.encoder_embed_dim,
            dropout=args.attention_dropout,
            encoder_decoder_attention=True,
        )

        self.encoder_attn_layer_norm = LayerNorm(self.embed_dim)
        self.fc1 = generate_linear(self.embed_dim, args.decoder_ffn_embed_dim)
        self.fc2 = generate_linear(args.decoder_ffn_embed_dim, self.embed_dim)
        self.final_layer_norm = LayerNorm(self.embed_dim)
        self.need_attn = True

    def forward(self,
                state,
                encoder_out=None,
                encoder_padding_mask=None,
                incremental_state=None,
                prev_self_attn_state=None,
                self_attn_mask=None,
                self_attn_padding_mask=None,
                need_attn=False,
                need_head_weights=False):
        """Forward pass of a single Transformer Decoder Layer"""

        # need_attn must be True if need_head_weights
        need_attn = True if need_head_weights else need_attn

        residual = state.clone()
        state, _ = self.self_attn(query=state,
                                  key=state,
                                  value=state,
                                  key_padding_mask=self_attn_padding_mask,
                                  need_weights=False,
                                  attn_mask=self_attn_mask)
        state = F.dropout(state, p=self.dropout, training=self.training)
        state += residual
        state = self.self_attn_layer_norm(state)

        residual = state.clone()
        '''
        ___QUESTION-6-DESCRIBE-E-START___
        1.  Add tensor shape annotation to EVERY TENSOR below (NOT just the output tensor)
        2.  How does encoder attention differ from self attention? 
        3.  What is the difference between key_padding_mask and attn_mask? 
        4.  If you understand this difference, then why don't we need to give attn_mask here?
        '''
        
        state, attn = self.encoder_attn(query=state,
                                        key=encoder_out,
                                        value=encoder_out,
                                        key_padding_mask=encoder_padding_mask,
                                        need_weights=need_attn or (not self.training and self.need_attn))
        
        # state.size = [target sentence length, batch of sentences, embedding features]
        # attn.size = [number of attention heads, batch of sentences, target sentence length, head embedding features]
        # encoder_out.size = [target sentence length, batch of sentences, embedding features]
        # encoder_padding_mask.size = [batch of sentences, target sentence length]
        #2. The encoder attention takes the key and value pairs from the encoder output and the query vectors from the
        #   decoder self-attention output. Basically, the encoder attention applies the attention mechanism on the
        #   decoder target words onto the encoder source words. In other words, how related are the target words with the source
        #   words. Strictically speaking this is not a self-attention mechanism, but an attention mechanism between a source
        #   and target sentence.
        #3. Key_padding_mask contains the masking for the sentence padding, that is, sentences that are shorter than the
        #   largest sentence in the batch may pad with zero values the remaining timesteps. On the other hand attn_mask
        #   is used to mask the self-attention cell in the decoder. This is done to prevent timesteps to attending to future
        #   timesteps: predictions for timestep t depend only on known outputs before time t. While key_padding_mask is represented
        #   as boolean vectors of size max_sentence_length for every sentence in a batch, where True means sentence padding is
        #   applied in that given word position, attn_mask is a square matrix with the upper triangle filled with -Inf values.
        #   This is done so every word in the first dimension only is able to look to itself and the previous words.
        #4. Since here we are not doing self-attention, there is no risk of future words information leaking into the attention
        #   cell. We are comparing each encoder word (query) against the whole source (key and values) which we do have. In
        #   decoder self-attention we are comparing each decoder target word with the whole target sentence, which we only have
        #   up until the current timestep target word.
        
        '''
        ___QUESTION-6-DESCRIBE-E-END___
        '''

        state = F.dropout(state, p=self.dropout, training=self.training)
        state += residual
        state = self.encoder_attn_layer_norm(state)

        residual = state.clone()
        state = F.relu(self.fc1(state))
        state = F.dropout(state, p=self.activation_dropout, training=self.training)
        state = self.fc2(state)
        state = F.dropout(state, p=self.dropout, training=self.training)
        state += residual
        state = self.final_layer_norm(state)

        return state, attn


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention"""
    def __init__(self,
                 embed_dim,
                 num_attn_heads,
                 kdim=None,
                 vdim=None,
                 dropout=0.,
                 self_attention=False,
                 encoder_decoder_attention=False):
        '''
        ___QUESTION-7-MULTIHEAD-ATTENTION-NOTE
        You shouldn't need to change the __init__ of this class for your attention implementation
        '''
        super().__init__()
        self.embed_dim = embed_dim
        self.k_embed_size = kdim if kdim else embed_dim
        self.v_embed_size = vdim if vdim else embed_dim

        self.num_heads = num_attn_heads
        self.attention_dropout = dropout
        self.head_embed_size = embed_dim // num_attn_heads  # this is d_k in the paper
        self.head_scaling = math.sqrt(self.head_embed_size)

        self.self_attention = self_attention
        self.enc_dec_attention = encoder_decoder_attention

        kv_same_dim = self.k_embed_size == embed_dim and self.v_embed_size == embed_dim
        assert self.head_embed_size * self.num_heads == self.embed_dim, "Embed dim must be divisible by num_heads!"
        assert not self.self_attention or kv_same_dim, "Self-attn requires query, key and value of equal size!"
        assert self.enc_dec_attention ^ self.self_attention, "One of self- or encoder- attention must be specified!"

        self.k_proj = nn.Linear(self.k_embed_size, embed_dim, bias=True)
        self.v_proj = nn.Linear(self.v_embed_size, embed_dim, bias=True)
        self.q_proj = nn.Linear(self.k_embed_size, embed_dim, bias=True)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)

        # Xavier initialisation
        nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.)

    def forward(self,
                query,
                key,
                value,
                key_padding_mask=None,
                attn_mask=None,
                need_weights=True):

        # Get size features
        tgt_time_steps, batch_size, embed_dim = query.size()
        assert self.embed_dim == embed_dim

        '''
        ___QUESTION-7-MULTIHEAD-ATTENTION-START
        Implement Multi-Head attention  according to Section 3.2.2 of https://arxiv.org/pdf/1706.03762.pdf.
        Note that you will have to handle edge cases for best model performance. Consider what behaviour should
        be expected if attn_mask or key_padding_mask are given?
        '''

        # attn is the output of MultiHead(Q,K,V) in Vaswani et al. 2017
        # attn must be size [tgt_time_steps, batch_size, embed_dim]
        # attn_weights is the combined output of h parallel heads of Attention(Q,K,V) in Vaswani et al. 2017
        # attn_weights must be size [num_heads, batch_size, tgt_time_steps, key.size(0)]
        # TODO: REPLACE THESE LINES WITH YOUR IMPLEMENTATION ------------------------ CUT

        

        attn = torch.zeros(size=(tgt_time_steps, batch_size, embed_dim))
        #print(key.transpose(0,1).transpose(1,2).size())
        
        
        # Query, key, and value projections
        query = self.q_proj(query) # query.size = [sentence length, batch of sentences, embedding features]
        key = self.k_proj(key) # key.size = [sentence length, batch of sentences, embedding features]
        value = self.v_proj(value) # value.size = [sentence length, batch of sentences, embedding features]
        
        
        # preparing query, key and value matrices for dot product and attention calculation
        query = query.unsqueeze(2) # query.size = [sentence length, batch of sentences, dimension placeholder for num_heads,embedding features]
        query = query.transpose(0,1) # query.size = [batch of sentences, sentence length, dimension placeholder for num_heads,embedding features]
        

        key = key.unsqueeze(2) # key.size = [sentence length, batch of sentences, dimension placeholder for num_heads,embedding features]
        key = key.transpose(0,1) # key.size = [batch of sentences, sentence length, dimension placeholder for num_heads,embedding features]

        value = value.unsqueeze(2) # value.size = [sentence length, batch of sentences, dimension placeholder for num_heads,embedding features]
        value = value.transpose(0,1) # value.size = [batch of sentences, sentence length, dimension placeholder for num_heads,embedding features]
        
        
        
        # Multi-head attention: assigning heads to inputs for narrow self-attention
        # Preparing each head for separate dot product and softmax operations
        query = query.view(batch_size,-1,self.num_heads,self.head_embed_size) # query.size = [batch of sentences, sentence length, num_heads,embedding features]
        query = query.transpose(1,2) # query.size = [[batch of sentences, num_heads, sentence length, embedding features]]
        
        
        key = key.view(batch_size,-1,self.num_heads,self.head_embed_size) # key.size = [batch of sentences, sentence length, num_heads,embedding features]
        key = key.transpose(1,2) # key.size = [[batch of sentences, num_heads, sentence length, embedding features]]

        
        value = value.view(batch_size,-1,self.num_heads,self.head_embed_size) # value.size = [batch of sentences, sentence length, num_heads,embedding features]
        value = value.transpose(1,2) # value.size = [[batch of sentences, num_heads, sentence length, embedding features]]



        
        
        score = torch.matmul(query,key.transpose(-2,-1))/self.head_scaling # score.size = [batch of sentences, number of heads, query sentence length, key sentence length]
        
        if attn_mask is not None:
            # If the element of the mask is 0, we shut 'off'. This is because for y1, we only want to look at x1, for y2 we only look at
            # x1 and x2, for y3 we look at x1, x2,x3 and so... If we don't do that, then we are essentially learning a mapping from
            # input to output.
            # Add to score the attn_mask to get -Inf values for future timesteps and not change the values for past timesteps.
            score += attn_mask.unsqueeze(0).unsqueeze(1) # score.size = [batch of sentences, number of heads, query sentence length, key sentence length]
            
            

        
        if key_padding_mask is not None:
            
            # Prepare key_padding_mask: will use element-wise multiplication to get the word to word matrix previous to masking.
            key_padding_mask = torch.logical_not(key_padding_mask) # key_padding_mask.size = [batch of sentences, sentence length]
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2) # key_padding_mask.size = [batch of sentences, dimension placeholder, dimension placeholder, sentence length]
            
            # Transposing masking to later element-wise multiply and get the final masking matrix (0 in rows and columns for padded words)
            
            # Perform masking filling with very low values to force softmax to assign 0 probability.
            score = score.masked_fill(key_padding_mask==False,float('-inf')) # score.size = [batch of sentences, number of heads, query sentence length, key sentence length]
            
        
        alpha=F.softmax(score,dim=-1) # alpha.size = [batch of sentences, number of heads, query sentence length, key sentence length]
        
        
        
        
        if self.attention_dropout is not None:
            
            alpha = F.dropout(alpha,self.attention_dropout) # alpha.size = [batch of sentences, number of heads, query sentence length, key sentence length]
        
        attention = torch.matmul(alpha,value) # attention.size = [batch of sentences, number of heads, query sentence length, head embedding features]
        
        attention_t = attention.transpose(1,2) # attention_t.size = [batch of sentences, query sentence length, number of heads, head embedding features]
        
        attention_t = attention_t.transpose(0,1) # attention_t.size = [query sentence length, batch of sentences, number of heads, head embedding features]
        
        attn = attention_t.contiguous().view(-1,batch_size,self.num_heads*self.head_embed_size) #attn.size = [query sentence length, batch of sentences, embedding features]
        

        attn = self.out_proj(attn) #attn.size = [query sentence length, batch of sentences, embedding features]
        
        attn_weights = attention.transpose(0,1) if need_weights else None #attn_weights.size = [number of heads, batch of sentences, target sentence length, head embedding features]
        
        
        #attn_weights = torch.zeros(size=(self.num_heads, batch_size, tgt_time_steps, -1)) if need_weights else None
        # TODO: --------------------------------------------------------------------- CUT

        '''
        ___QUESTION-7-MULTIHEAD-ATTENTION-END
        '''

        return attn, attn_weights


class PositionalEmbedding(nn.Module):
    def __init__(self, embed_dim, padding_idx, init_size=1024):
        super().__init__()
        self.embed_dim = embed_dim
        self.padding_idx = padding_idx
        self.weights = PositionalEmbedding.get_embedding(init_size, embed_dim, padding_idx)
        self.register_buffer("_float_tensor", torch.FloatTensor(1))

    @staticmethod
    def get_embedding(num_embeddings, embed_dim, padding_idx=None):
        half_dim = embed_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if embed_dim % 2 == 1:
            # Zero pad in specific mismatch case
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0.
        return emb

    def forward(self, inputs, incremental_state=None, timestep=None):
        batch_size, seq_len = inputs.size()
        max_pos = self.padding_idx + 1 + seq_len

        if self.weights is None or max_pos > self.weights.size(0):
            # Expand embeddings if required
            self.weights = PositionalEmbedding.get_embedding(max_pos, self.embed_dim, self.padding_idx)
        self.weights = self.weights.to(self._float_tensor)

        if incremental_state is not None:
            #   Positional embed is identical for all tokens during single step decoding
            pos = timestep.view(-1)[0] + 1 if timestep is not None else seq_len
            return self.weights.index_select(index=self.padding_idx + pos, dim=0).unsqueeze(1).repeat(batch_size, 1, 1)

        # Replace non-padding symbols with position numbers from padding_idx+1 onwards.
        mask = inputs.ne(self.padding_idx).int()
        positions = (torch.cumsum(mask, dim=1).type_as(inputs) * mask).long() + self.padding_idx

        # Lookup positional embeddings for each position and return in shape of input tensor w/o gradient
        return self.weights.index_select(0, positions.view(-1)).view(batch_size, seq_len, -1).detach()


def LayerNorm(normal_shape, eps=1e-5):
    return torch.nn.LayerNorm(normalized_shape=normal_shape, eps=eps, elementwise_affine=True)


def fill_with_neg_inf(t):
    return t.float().fill_(float('-inf')).type_as(t)


def generate_embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def generate_linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m

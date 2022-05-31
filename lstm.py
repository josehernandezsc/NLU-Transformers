import torch
import torch.nn as nn
import torch.nn.functional as F

from seq2seq import utils
from seq2seq.models import Seq2SeqModel, Seq2SeqEncoder, Seq2SeqDecoder
from seq2seq.models import register_model, register_model_architecture


@register_model('lstm')
class LSTMModel(Seq2SeqModel):
    """ Defines the sequence-to-sequence model class. """

    def __init__(self,
                 encoder,
                 decoder):

        super().__init__(encoder, decoder)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument('--encoder-embed-dim', type=int, help='encoder embedding dimension')
        parser.add_argument('--encoder-embed-path', help='path to pre-trained encoder embedding')
        parser.add_argument('--encoder-hidden-size', type=int, help='encoder hidden size')
        parser.add_argument('--encoder-num-layers', type=int, help='number of encoder layers')
        parser.add_argument('--encoder-bidirectional', help='bidirectional encoder')
        parser.add_argument('--encoder-dropout-in', help='dropout probability for encoder input embedding')
        parser.add_argument('--encoder-dropout-out', help='dropout probability for encoder output')

        parser.add_argument('--decoder-embed-dim', type=int, help='decoder embedding dimension')
        parser.add_argument('--decoder-embed-path', help='path to pre-trained decoder embedding')
        parser.add_argument('--decoder-hidden-size', type=int, help='decoder hidden size')
        parser.add_argument('--decoder-num-layers', type=int, help='number of decoder layers')
        parser.add_argument('--decoder-dropout-in', type=float, help='dropout probability for decoder input embedding')
        parser.add_argument('--decoder-dropout-out', type=float, help='dropout probability for decoder output')
        parser.add_argument('--decoder-use-attention', help='decoder attention')
        parser.add_argument('--decoder-use-lexical-model', help='toggle for the lexical model')

    @classmethod
    def build_model(cls, args, src_dict, tgt_dict):
        """ Constructs the model. """
        base_architecture(args)
        encoder_pretrained_embedding = None
        decoder_pretrained_embedding = None

        # Load pre-trained embeddings, if desired
        if args.encoder_embed_path:
            encoder_pretrained_embedding = utils.load_embedding(args.encoder_embed_path, src_dict)
        if args.decoder_embed_path:
            decoder_pretrained_embedding = utils.load_embedding(args.decoder_embed_path, tgt_dict)

        # Construct the encoder
        encoder = LSTMEncoder(dictionary=src_dict,
                              embed_dim=args.encoder_embed_dim,
                              hidden_size=args.encoder_hidden_size,
                              num_layers=args.encoder_num_layers,
                              bidirectional=args.encoder_bidirectional,
                              dropout_in=args.encoder_dropout_in,
                              dropout_out=args.encoder_dropout_out,
                              pretrained_embedding=encoder_pretrained_embedding)

        # Construct the decoder
        decoder = LSTMDecoder(dictionary=tgt_dict,
                              embed_dim=args.decoder_embed_dim,
                              hidden_size=args.decoder_hidden_size,
                              num_layers=args.decoder_num_layers,
                              dropout_in=args.decoder_dropout_in,
                              dropout_out=args.decoder_dropout_out,
                              pretrained_embedding=decoder_pretrained_embedding,
                              use_attention=bool(eval(args.decoder_use_attention)),
                              use_lexical_model=bool(eval(args.decoder_use_lexical_model)))
        return cls(encoder, decoder)


class LSTMEncoder(Seq2SeqEncoder):
    """ Defines the encoder class. """

    def __init__(self,
                 dictionary,
                 embed_dim=64,
                 hidden_size=64,
                 num_layers=1,
                 bidirectional=True,
                 dropout_in=0.25,
                 dropout_out=0.25,
                 pretrained_embedding=None):

        super().__init__(dictionary)
        self.num_layers = num_layers
        self.dropout_in = dropout_in
        self.dropout_out = dropout_out
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.output_dim = 2 * hidden_size if bidirectional else hidden_size

        if pretrained_embedding is not None:
            self.embedding = pretrained_embedding
        else:
            self.embedding = nn.Embedding(len(dictionary), embed_dim, dictionary.pad_idx)

        dropout_lstm = dropout_out if num_layers > 1 else 0.
        self.lstm = nn.LSTM(input_size=embed_dim,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            dropout=dropout_lstm,
                            bidirectional=bool(bidirectional))
    
    def forward(self, src_tokens, src_lengths):
        
        """ Performs a single forward pass through the instantiated encoder sub-network. """
        # Embed tokens and apply dropout
        batch_size, src_time_steps = src_tokens.size()
        
        src_embeddings = self.embedding(src_tokens)
        
        _src_embeddings = F.dropout(src_embeddings, p=self.dropout_in, training=self.training)

        # Transpose batch: [batch_size, src_time_steps, num_features] -> [src_time_steps, batch_size, num_features]
        src_embeddings = _src_embeddings.transpose(0, 1) # src_embeddings (num palabras, numero de sentences, embedding)
        
        # Pack embedded tokens into a PackedSequence - SRC_LENGTHS es la lista con la maxima longitud de palabras por batch
        packed_source_embeddings = nn.utils.rnn.pack_padded_sequence(src_embeddings, src_lengths.data.tolist())
        # concatena cada oración luego de hacer el padding - (num_palabras * 10, 64)
        
        # Pass source input through the recurrent layer(s)
        if self.bidirectional:
            state_size = 2 * self.num_layers, batch_size, self.hidden_size
        else:
            state_size = self.num_layers, batch_size, self.hidden_size

        hidden_initial = src_embeddings.new_zeros(*state_size)
        #print(hidden_initial.size())
        context_initial = src_embeddings.new_zeros(*state_size)
        
        packed_outputs, (final_hidden_states, final_cell_states) = self.lstm(packed_source_embeddings,
                                                                             (hidden_initial, context_initial))
        
        # Unpack LSTM outputs and optionally apply dropout (dropout currently disabled)
        lstm_output, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, padding_value=0.)
        lstm_output = F.dropout(lstm_output, p=self.dropout_out, training=self.training)
        #print(final_hidden_states.size())
        
        assert list(lstm_output.size()) == [src_time_steps, batch_size, self.output_dim]  # sanity check

        '''
        ___QUESTION-1-DESCRIBE-A-START___
        1.  Add tensor shape annotation to each of the output tensor
            output tensor means tensors on the left hand side of "="
            e.g., 
                sent_tensor = create_sentence_tensor(...) 
                # sent_tensor.size = [batch, sent_len, hidden]
        2.  Describe what happens when self.bidirectional is set to True. 
        3.  What is the difference between final_hidden_states and final_cell_states?
        '''
        if self.bidirectional:
            def combine_directions(outs):
                return torch.cat([outs[0: outs.size(0): 2], outs[1: outs.size(0): 2]], dim=2)
            final_hidden_states = combine_directions(final_hidden_states)
            # final_hidden_states.size = [1, 10, 128]
            # final_hidden_states.size = [num_layers, batch_size: batch_of_sentences, output_size: embedding_size]
            final_cell_states = combine_directions(final_cell_states)
            # final_cell_states.size = [1, 10, 128]
            # final_hidden_states.size = [num_layers, batch_size: batch_of_sentences, output_size: embedding_size]

            # 2. The LSTM encoder uses a bidirectional mechanism to capture forward and backwards dependencies. 
            # The hidden and cell states of both RNNs (size: [2,10,64] i.e. [1,10,64] for each direction) are 
            # concatenated across the last dimension, resulting in tensors of shape [1,10,128]. The effect of this 
            # is that now, the output of both RNNs is 'flattened' into the last dimension, which is why the output 
            # size of the last dimension is 128 (64*2) similar as how it is in ELMo.
            # 3. The hidden states carries/produces the information for the current time-step, this is then passed 
            # onto the next time-step and used to produce the next hidden state. Hidden states are replaced on each 
            # time-step. Hence, final\_hidden\_states contains the information of the last hidden state, this corresponds 
            # to the embeddings of the model before going through the output layer. In the context of a MT model, the last 
            # hidden state contains the embeddings of the last translated token. On the other hand, the cell state is a 
            # long-term memory that stores information from previous time-steps, not necessarily the previous one, i.e. 
            # it is not replaced on each time-step. Hence, final\_cell\_states contains the long-term memory after going 
            # through all time-steps, which in fact, is a parameter used to compute the last hidden state.
        '''___QUESTION-1-DESCRIBE-A-END___'''

        # Generate mask zeroing-out padded positions in encoder inputs
        src_mask = src_tokens.eq(self.dictionary.pad_idx)
        
        #print(src_mask if src_mask.any() else None)
        return {'src_embeddings': _src_embeddings.transpose(0, 1),
                'src_out': (lstm_output, final_hidden_states, final_cell_states),
                'src_mask': src_mask if src_mask.any() else None}


class AttentionLayer(nn.Module):
    """ Defines the attention layer class. Uses Luong's global attention with the general scoring function. """
    def __init__(self, input_dims, output_dims):
        super().__init__()
        # Scoring method is 'general'
        
        self.src_projection = nn.Linear(input_dims, output_dims, bias=False)
        self.context_plus_hidden_projection = nn.Linear(input_dims + output_dims, output_dims, bias=False)

    def forward(self, tgt_input, encoder_out, src_mask):
        #print(encoder_out.size())
        # tgt_input has shape = [batch_size, input_dims]
        # encoder_out has shape = [src_time_steps, batch_size, output_dims]
        # src_mask has shape = [batch_size, src_time_steps]
        #print(tgt_input.size())
        # Get attention scores
        # [batch_size, src_time_steps, output_dims]
        encoder_out = encoder_out.transpose(1, 0)
        
        # [batch_size, 1, src_time_steps]
        #QUE ES TGT_INPUT?????
        
        
        attn_scores = self.score(tgt_input, encoder_out)
        
        '''
        ___QUESTION-1-DESCRIBE-B-START___
        1.  Add tensor shape annotation to each of the output tensor
            output tensor means tensors on the left hand side of "="
            e.g., 
                sent_tensor = create_sentence_tensor(...) 
                # sent_tensor.size = [batch, sent_len, hidden]
        2.  Describe how the attention context vector is calculated. 
        3.  Why do we need to apply a mask to the attention scores?
        '''
        #POR QUÉ HAY TAN POCOS MASKS???? Acaso las oraciones estan todas ordenadas
        if src_mask is not None:
            
            src_mask = src_mask.unsqueeze(dim=1)
            
            # src_mask.size = [batch_size, single_token, sentence_length]
            attn_scores.masked_fill_(src_mask, float('-inf'))
            
            # attn_scores.size = [batch_size, single_token, sentence_length]
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        #print(attn_weights.size())
        # attn_weights.size = [batch_size, single_token, sentence_length]
        attn_context = torch.bmm(attn_weights, encoder_out).squeeze(dim=1)
        #print(torch.bmm(attn_weights, encoder_out).size())
        # attn_context.size([batch_size, single_token, embedding_size])
        context_plus_hidden = torch.cat([tgt_input, attn_context], dim=1)
        # context_plus_hidden.size = [batch_size, input + context concatenated embedding]
        #print(context_plus_hidden.size())
        attn_out = torch.tanh(self.context_plus_hidden_projection(context_plus_hidden))
        # attn_out.size = [batch_size, attn_out embedding projection back to 128]
        
        #2. The attention context performs a matrix multiplication for each batch (each 10 sentences) between each of the
        #   attention weights previously calculated (obtained by the dot product between a target token and each token in
        #   a source sentence) and each of the tokens from the LSTM encoder for a given sentence. That is, the attention weights
        #   from each token in a sentence are multiplied with each embedding vector component coming from each token from 
        #   the decoder and summed up giving a context vector the same size as the embedding vector for each sentence.

        #3. Some sentences are needed to be padded with zeros to account for sentences with fewer tokens than the largest
        #   sentence to maintain a fixed shape per batch. For example, if 9 sentences have 12 tokens, and one sentence has 
        #   10 tokens, that sentence needs to be padded with 0s in its last two component. This is reflected in the output
        #   from the decoder which outputs 0s for these paddings. Our mask keeps information on which tokens are padded,
        #   which allows the attention weights to assign zero weights to these cases, since it doesn't hold any attention
        #   information, by setting these padded 0 values to -Inf. This is done so when the softmax outputs the attention
        #   weights, it outputs 0 for the -Inf values, which are our padding tokens.
        '''___QUESTION-1-DESCRIBE-B-END___'''

        return attn_out, attn_weights.squeeze(dim=1)

    def score(self, tgt_input, encoder_out):
        """ Computes attention scores. """

        '''
        ___QUESTION-1-DESCRIBE-C-START___
        1.  Add tensor shape annotation to each of the output tensor
        2.  How are attention scores calculated? 
        3.  What role does batch matrix multiplication (i.e. torch.bmm()) play in aligning encoder and decoder representations?
        '''
        
        projected_encoder_out = self.src_projection(encoder_out).transpose(2, 1)
        
        # projected_encoder_out.size = [batch of sentences, embedding, sentence length]
        attn_scores = torch.bmm(tgt_input.unsqueeze(dim=1), projected_encoder_out)
        # attn_scores.size = [batch of sentences, single token, sentence length]
        
        #2. The output from the encoder is projected in this case to the same dimensionality as the target input which
        #   represents every token from the target sentences (english translation). It is transposed so that we have a
        #   shape [batch of sentences, embedding, sentence length] and by doing a matrix multiplication per sentence
        #   (per batch), we can do a dot product of the target token by each of the source tokens across the 128-dimension
        #   embedding, endding with a weight representative of each token in the source sentence for a given target token.

        #3. torch.bmm helps us to perform several matrix multiplications across the first dimension which is the batch
        #   length. This can be interpreted as treating each sentence (batch element) separately and applying a matrix
        #   multiplication on each one, outputting the matrix multiplication results for each batch element.
        '''___QUESTION-1-DESCRIBE-C-END___'''

        return attn_scores


class LSTMDecoder(Seq2SeqDecoder):
    """ Defines the decoder class. """

    def __init__(self,
                 dictionary,
                 embed_dim=64,
                 hidden_size=128,
                 num_layers=1,
                 dropout_in=0.25,
                 dropout_out=0.25,
                 pretrained_embedding=None,
                 use_attention=True,
                 use_lexical_model=False):

        super().__init__(dictionary)

        self.dropout_in = dropout_in
        self.dropout_out = dropout_out
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size

        if pretrained_embedding is not None:
            self.embedding = pretrained_embedding
        else:
            self.embedding = nn.Embedding(len(dictionary), embed_dim, dictionary.pad_idx)

        # Define decoder layers and modules
        self.attention = AttentionLayer(hidden_size, hidden_size) if use_attention else None

        self.layers = nn.ModuleList([nn.LSTMCell(
            input_size=hidden_size + embed_dim if layer == 0 else hidden_size,
            hidden_size=hidden_size)
            for layer in range(num_layers)])
        
        self.final_projection = nn.Linear(hidden_size, len(dictionary))

        self.use_lexical_model = use_lexical_model





        #self.ffn=nn.Linear(embed_dim,embed_dim)
        

        if self.use_lexical_model:
            
            # __QUESTION-5: Add parts of decoder architecture corresponding to the LEXICAL MODEL here
            self.ffn=nn.Linear(embed_dim,embed_dim,bias=False)
            self.ffn2 = nn.Linear(embed_dim,len(dictionary))
            #pass
            # TODO: --------------------------------------------------------------------- /CUT

    def forward(self, tgt_inputs, encoder_out, incremental_state=None):
        """ Performs the forward pass through the instantiated model. """
        # Optionally, feed decoder input token-by-token
        if incremental_state is not None:
            tgt_inputs = tgt_inputs[:, -1:]
        
        # __QUESTION-5 : Following code is to assist with the LEXICAL MODEL implementation
        # Recover encoder input
        src_embeddings = encoder_out['src_embeddings']

        src_out, src_hidden_states, src_cell_states = encoder_out['src_out']
        
        src_mask = encoder_out['src_mask']
        src_time_steps = src_out.size(0)

        # Embed target tokens and apply dropout
        batch_size, tgt_time_steps = tgt_inputs.size()
        
        tgt_embeddings = self.embedding(tgt_inputs)
        tgt_embeddings = F.dropout(tgt_embeddings, p=self.dropout_in, training=self.training)

        # Transpose batch: [batch_size, tgt_time_steps, num_features] -> [tgt_time_steps, batch_size, num_features]
        tgt_embeddings = tgt_embeddings.transpose(0, 1)

        # Initialize previous states (or retrieve from cache during incremental generation)
        '''
        ___QUESTION-1-DESCRIBE-D-START___
        1.  Add tensor shape annotation to each of the output tensor
        2.  Describe how the decoder state is initialized. 
        3.  When is cached_state == None? 
        4.  What role does input_feed play?
        '''
        cached_state = utils.get_incremental_state(self, incremental_state, 'cached_state')
        
        if cached_state is not None:
            tgt_hidden_states, tgt_cell_states, input_feed = cached_state
        else:
            
            tgt_hidden_states = [torch.zeros(tgt_inputs.size()[0], self.hidden_size) for i in range(len(self.layers))]
            #tgt_hidden_states.size = [batch of sentences, embedding] - within a list for each layer
            tgt_cell_states = [torch.zeros(tgt_inputs.size()[0], self.hidden_size) for i in range(len(self.layers))]
            #tgt_cell_states.size = [batch of sentences, embedding] - within a list for each layer
            input_feed = tgt_embeddings.data.new(batch_size, self.hidden_size).zero_()
            #input_feed.size = [batch of sentences, embedding] - within a list for each layer

        #2. The decoder state initialization depends on whether we are using an incremental approach or not. In the default
        #   setting we are not using an incremental state, thus the hidden and cell states, and the input feed are initiliazed
        #   as tensor of zeros for each sentence and embedding feature. This is common since for our first token we don't
        #   have a previous state or memory cell and our input feed will just be just the first token as in tgt_embeddings.
        #   If an incremental approach is used, each token is fed one by one on each forward pass, thus we need to remember
        #   our previous states and cells and lstm output as well for the next timestep.
        #3. Cached_state is equal to none when no incremental approach is taken. In this case, we process our whole sentence
        #   through all the timesteps in a single forward pass. Cached state is used when doing inference.
        #4. Input feed gives the output from the previous timestep's hidden state in the last layer, that is, our input for
        #   our next recurrent step coming from our previous timestep, which is concatenated to our current step token input.
        #   For incremental approaches we need to save the input feed so we can provide our network with this as the previous
        #   final layer hidden state for the next forward pass.
        '''___QUESTION-1-DESCRIBE-D-END___'''
        
        # Initialize attention output node
        attn_weights = tgt_embeddings.data.new(batch_size, tgt_time_steps, src_time_steps).zero_()
        rnn_outputs = []

        # __QUESTION-5 : Following code is to assist with the LEXICAL MODEL implementation
        # Cache lexical context vectors per translation time-step
        lexical_contexts = []
        
        for j in range(tgt_time_steps):
            # Concatenate the current token embedding with output from previous time step (i.e. 'input feeding')
            lstm_input = torch.cat([tgt_embeddings[j, :, :], input_feed], dim=1)

            for layer_id, rnn_layer in enumerate(self.layers):
                
                # Pass target input through the recurrent layer(s)
                tgt_hidden_states[layer_id], tgt_cell_states[layer_id] = \
                    rnn_layer(lstm_input, (tgt_hidden_states[layer_id], tgt_cell_states[layer_id]))

                # Current hidden state becomes input to the subsequent layer; apply dropout
                
                
                lstm_input = F.dropout(tgt_hidden_states[layer_id], p=self.dropout_out, training=self.training)
                
                
            '''
            ___QUESTION-1-DESCRIBE-E-START___
            1.  Add tensor shape annotation to each of the output tensor
            2.  How is attention integrated into the decoder? 
            3.  Why is the attention function given the previous target state as one of its inputs? 
            4.  What is the purpose of the dropout layer?
            '''
            
            if self.attention is None:
                input_feed = tgt_hidden_states[-1]
                #input_feed.size = [batch of sentences, embedding features]
            else:
                
                input_feed, step_attn_weights = self.attention(tgt_hidden_states[-1], src_out, src_mask)
                #input_feed.size = [batch of sentences, embedding features]
                #step_attn_weights.size = [batch of sentences, source sentence length]
                
                attn_weights[:, j, :] = step_attn_weights

                
                #attn_weights.size = [batch of sentences, target sentence length, source sentence length]
                

                
                
                if self.use_lexical_model:
                    # __QUESTION-5: Compute and collect LEXICAL MODEL context vectors here
                    
                    step_attn_weights=step_attn_weights.unsqueeze(1)
                    
                    src_embeddings_lex = src_embeddings.transpose(0,1)
                    
                    lexical_contexts.append(torch.bmm(step_attn_weights,src_embeddings_lex))
                    
                    
                    #pass
                    # TODO: --------------------------------------------------------------------- /CUT
            
            input_feed = F.dropout(input_feed, p=self.dropout_out, training=self.training)
            rnn_outputs.append(input_feed)
            #2. Attention takes as inputs the output from the decoder containing the embeddings for all the tokens in
            #   a sentence, in batches of 10 sentences. Those would be the attention keys. Additionally it takes the output
            #   from each target hidden state last layer (attention query which acts as the input to the next timestep in the LSTM) and performs the
            #   attention scoring with the keys. Finally the resulting embedding for the current timestep token under the
            #   attention mechanism is output.
            #3. The previous target state will be the query for which the attention mechanism will compare against all
            #   the encoder output token embeddings (keys). This way we can model the attention that each target token
            #   takes on every source token (as encoder outputs).
            #4. We apply dropout to reduce overfitting and improve model generalization by regularizing it.
            '''___QUESTION-1-DESCRIBE-E-END___'''
        
        # Cache previous states (only used during incremental, auto-regressive generation)
        
        utils.set_incremental_state(
            self, incremental_state, 'cached_state', (tgt_hidden_states, tgt_cell_states, input_feed))

        # Collect outputs across time steps
        decoder_output = torch.cat(rnn_outputs, dim=0).view(tgt_time_steps, batch_size, self.hidden_size)

        # Transpose batch back: [tgt_time_steps, batch_size, num_features] -> [batch_size, tgt_time_steps, num_features]
        decoder_output = decoder_output.transpose(0, 1)
        
        # Final projection
        decoder_output = self.final_projection(decoder_output)
        



        
        #print(len(lexical_contexts[0]))
        
        

        if self.use_lexical_model:
            # __QUESTION-5: Incorporate the LEXICAL MODEL into the prediction of target tokens here
            sum_embed=torch.cat(lexical_contexts,dim=1)
            #sum_embed=sum_embed.view(len(lexical_contexts[0]),len(lexical_contexts),-1)
        
            f_t=torch.tanh(sum_embed)
            fnn=torch.tanh(self.ffn(f_t))+f_t
            decoder_output = decoder_output+self.ffn2(fnn)
            #pass
            # TODO: --------------------------------------------------------------------- /CUT
        
        return decoder_output, attn_weights


@register_model_architecture('lstm', 'lstm')
def base_architecture(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 64)
    args.encoder_embed_path = getattr(args, 'encoder_embed_path', None)
    args.encoder_hidden_size = getattr(args, 'encoder_hidden_size', 64)
    args.encoder_num_layers = getattr(args, 'encoder_num_layers', 1)
    args.encoder_bidirectional = getattr(args, 'encoder_bidirectional', 'True')
    args.encoder_dropout_in = getattr(args, 'encoder_dropout_in', 0.25)
    args.encoder_dropout_out = getattr(args, 'encoder_dropout_out', 0.25)

    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 64)
    args.decoder_embed_path = getattr(args, 'decoder_embed_path', None)
    args.decoder_hidden_size = getattr(args, 'decoder_hidden_size', 128)
    args.decoder_num_layers = getattr(args, 'decoder_num_layers', 1)
    args.decoder_dropout_in = getattr(args, 'decoder_dropout_in', 0.25)
    args.decoder_dropout_out = getattr(args, 'decoder_dropout_out', 0.25)
    args.decoder_use_attention = getattr(args, 'decoder_use_attention', 'True')
    args.decoder_use_lexical_model = getattr(args, 'decoder_use_lexical_model', 'False')

from individual import individual

class Population:
    def __init__(self, pop_size, min_layer, max_layer, input_width, input_height, input_channels, conv_prob, pool_prob, fc_prob, max_conv_kernel, max_out_ch, max_fc_neurons, output_dim):
        
        max_pool_layers = 0
        in_w = input_width

        while in_w > 4:
            max_pool_layers += 1
            in_w = in_w/2

        self.individual = []
        for i in range(pop_size):
            self.individual.append(individual(min_layer, max_layer, max_pool_layers, input_width, input_height, input_channels, conv_prob, pool_prob, fc_prob, max_conv_kernel, max_out_ch, max_fc_neurons, output_dim))

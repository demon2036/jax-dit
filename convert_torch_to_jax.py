import flax


def convert_torch_to_jax(state_dict):
    print(state_dict.keys())
    params = {
        'pos_embed': state_dict['pos_embed'],

        'x_embedder.proj.kernel': state_dict['x_embedder.proj.weight'].permute(2, 3, 1, 0),
        'x_embedder.proj.bias': state_dict['x_embedder.proj.bias'],

        't_embedder.mlp.layers_0.kernel': state_dict['t_embedder.mlp.0.weight'].transpose(1, 0),
        't_embedder.mlp.layers_0.bias': state_dict['t_embedder.mlp.0.bias'],
        't_embedder.mlp.layers_2.kernel': state_dict['t_embedder.mlp.2.weight'].transpose(1, 0),
        't_embedder.mlp.layers_2.bias': state_dict['t_embedder.mlp.2.bias'],

        'y_embedder.embedding_table.embedding': state_dict['y_embedder.embedding_table.weight'],

        'final_layer.adaLN_modulation.layers_1.kernel': state_dict['final_layer.adaLN_modulation.1.weight'].transpose(1,
                                                                                                                      0),
        'final_layer.adaLN_modulation.layers_1.bias': state_dict['final_layer.adaLN_modulation.1.bias'],
        'final_layer.linear.kernel': state_dict['final_layer.linear.weight'].transpose(1, 0),
        'final_layer.linear.bias': state_dict['final_layer.linear.bias'],

    }

    layer_idx = 0
    while f"blocks.{layer_idx}.attn.qkv.bias" in state_dict:
        attn_w_qkv = state_dict[f'blocks.{layer_idx}.attn.qkv.weight'].transpose(1, 0)
        attn_w_proj = state_dict[f'blocks.{layer_idx}.attn.proj.weight'].transpose(1, 0)
        attn_bias_qkv = state_dict[f'blocks.{layer_idx}.attn.qkv.bias']
        attn_bias_proj = state_dict[f'blocks.{layer_idx}.attn.proj.bias']

        params[f'blocks_{layer_idx}.attn.qkv.kernel'] = attn_w_qkv
        params[f'blocks_{layer_idx}.attn.proj.kernel'] = attn_w_proj
        params[f'blocks_{layer_idx}.attn.qkv.bias'] = attn_bias_qkv
        params[f'blocks_{layer_idx}.attn.proj.bias'] = attn_bias_proj

        mlp_w_fc1 = state_dict[f'blocks.{layer_idx}.mlp.fc1.weight'].transpose(1, 0)
        mlp_w_fc2 = state_dict[f'blocks.{layer_idx}.mlp.fc2.weight'].transpose(1, 0)
        mlp_bias_fc1 = state_dict[f'blocks.{layer_idx}.mlp.fc1.bias']
        mlp_bias_fc2 = state_dict[f'blocks.{layer_idx}.mlp.fc2.bias']

        params[f'blocks_{layer_idx}.mlp.fc1.kernel'] = mlp_w_fc1
        params[f'blocks_{layer_idx}.mlp.fc2.kernel'] = mlp_w_fc2
        params[f'blocks_{layer_idx}.mlp.fc1.bias'] = mlp_bias_fc1
        params[f'blocks_{layer_idx}.mlp.fc2.bias'] = mlp_bias_fc2

        adaLN_modulation_layers_1_kernel = state_dict[f'blocks.{layer_idx}.adaLN_modulation.1.weight'].transpose(1, 0)
        adaLN_modulation_layers_1_bias = state_dict[f'blocks.{layer_idx}.adaLN_modulation.1.bias']

        params[f'blocks_{layer_idx}.adaLN_modulation.layers_1.kernel'] = adaLN_modulation_layers_1_kernel
        params[f'blocks_{layer_idx}.adaLN_modulation.layers_1.bias'] = adaLN_modulation_layers_1_bias

        layer_idx += 1


    params = {k: v.numpy() for k, v in params.items()}

    params = flax.traverse_util.unflatten_dict(params, '.')


    return params

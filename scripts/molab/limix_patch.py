"""LimiX-2M ckpt config translation (old ckpt schema -> current main code).
Patches a COPY; weights untouched. Then trial-loads with strict=True and
reports missing/unexpected keys.
"""
import torch, sys, copy
sys.path.insert(0, '/marimo/limix')

SRC = '/tmp/limix_cache/models--stable-ai--LimiX-2M/snapshots/641d8b81e1c8b1b0e51cb4ae47de22c4844d7a1e/LimiX-2M.ckpt'
DST = '/tmp/limix_cache/LimiX-2M-v101.ckpt'

ckpt = torch.load(SRC, map_location='cpu', weights_only=False)
cfg = copy.deepcopy(ckpt['config'])

cfg['tf_mlp_layer_type'] = 'normal'
cfg['tf_mlp_activation_fuction'] = 'gelu'
cfg['tf_mlp_use_bias'] = False
cfg['tf_attention_layer_type'] = 'original'
cfg['tf_attention_use_bias'] = False
cfg['num_buckets'] = 0

N = cfg['nlayers']
cfg['model_structure_config'] = {
    'nlayers': N,
    'layers': [{'emsize': cfg['embed_dim'], 'nhead': cfg['nhead'],
                'hid_dim': cfg['hid_dim'], 'arch': 'smf'} for _ in range(N)],
}

rbf = cfg['encoder_config_x']['RBF_config']
rbf['RBF_token_embed_dim'] = rbf.pop('token_embed_dim')
rbf['RBF_n_kernels'] = rbf.pop('n_kernels')
rbf['RBF_sigma'] = rbf.pop('sigma')
rbf['RBF_use_learn_embeddings'] = rbf.pop('use_learn_embeddings')
rbf['RBF_exponent_digits'] = 1
rbf['RBF_log_base'] = 10

cfg['encoder_config_y']['num_features'] = cfg['encoder_config_y']['num_inputs']
cfg['encoder_config_x']['categorical_features_class_num'] = 100

ckpt['config'] = cfg
torch.save(ckpt, DST)
print('patched copy saved', DST, flush=True)

from model.v1_0.loading import build_model
model = build_model(cfg)
print('model built', flush=True)
missing, unexpected = model.load_state_dict(ckpt['state_dict'], strict=False)
print('MISSING:', len(missing))
for k in missing[:20]:
    print('  M-', k)
print('UNEXPECTED:', len(unexpected))
for k in unexpected[:20]:
    print('  U-', k)

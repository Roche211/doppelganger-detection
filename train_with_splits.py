import sys
sys.path.append('/workspace/src')

from training.train import SimilarityTrainer
from training.config import get_config
from data.dataset import get_data_loaders_from_splits

sllfw_split='/workspace/data/processed/metadata/sllfw_splits.json'
hda_split ='/workspace/data/processed/metadata/hda_splits.json'

train_loader, val_loader, test_loader = get_data_loaders_from_splits(
    processed_data_dir='/workspace/data/processed',
    batch_size=64, num_workers=4,
    sllfw_split_file=sllfw_split, hda_split_file=hda_split
)

cfg = get_config({
  'num_epochs': 40,            # freeze 20, then unfreeze
  'batch_size': 64,
  'learning_rate': 1e-4,       # head phase
  'freeze_backbone': True,
  'unfreeze_epoch': 20,        # then unfreeze
  'output_dir': '/workspace/models/checkpoints',
  'data_dir': '/workspace/data/processed',
  'log_frequency': 20,
})

trainer = SimilarityTrainer(cfg)
trainer.train(train_loader, val_loader)

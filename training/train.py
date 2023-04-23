#hide
from fastbook import *
from fastai.vision.widgets import *
block = DataBlock(
    blocks=(ImageBlock, CategoryBlock), 
    get_items=get_image_files, 
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=Resize(128))
# clocks = clocks.new(item_tfms=Resize(128), batch_tfms=aug_transforms(mult=2))
block = block.new(
    item_tfms=RandomResizedCrop(224, min_scale=0.5),
    batch_tfms=aug_transforms())

path = Path('family')
dls = block.dataloaders(path)
learn = vision_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(20)

learn.export()
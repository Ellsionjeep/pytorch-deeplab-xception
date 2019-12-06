from utils.dataloader import pascal
from torch.utils.data import DataLoader

def make_data_loader(args, **kwargs):

    if args.train:
        train_set = pascal.VOCSegmentation(args, split='train')
        val_set = pascal.VOCSegmentation(args, split='val')
        
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)

        return train_loader, val_loader
    
    elif args.inference:
        inference_set = pascal.VOCSegmentation(args, split='inference')
        inferenc_loader = DataLoader(inference_set, batch_size=1, shuffle=False)
        ids = inference_set.im_ids
        return inferenc_loader, ids

    else:
        raise NotImplementedError


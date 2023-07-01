from a5_helper import decode_captions

def get_data():
    import multiprocessing

    # Set a few constants related to data loading.
    IMAGE_SHAPE = (112, 112)
    NUM_WORKERS = multiprocessing.cpu_count()
    
    # Batch size used for full training runs:
    BATCH_SIZE = 256
    
    # Batch size used for overfitting sanity checks:
    OVR_BATCH_SIZE = BATCH_SIZE // 8
    
    # Batch size used for visualization:
    VIS_BATCH_SIZE = 4

    # Download and load serialized COCO data from coco.pt
    # It contains a dictionary of
    # "train_images" - resized training images (IMAGE_SHAPE)
    # "val_images" - resized validation images (IMAGE_SHAPE)
    # "train_captions" - tokenized and numericalized training captions
    # "val_captions" - tokenized and numericalized validation captions
    # "vocab" - caption vocabulary, including "idx_to_token" and "token_to_idx"
    
    if os.path.isfile("./datasets/coco.pt"):
        print("COCO data exists!")
    else:
        print("downloading COCO dataset")
        !wget http://web.eecs.umich.edu/~justincj/teaching/eecs498/coco.pt -P ./datasets/
    
    # load COCO data from coco.pt, loaf_COCO is implemented in a5_helper.py
    data_dict = load_coco_captions(path="./datasets/coco.pt")
    
    num_train = data_dict["train_images"].size(0)
    num_val = data_dict["val_images"].size(0)
    
    # declare variables for special tokens
    NULL_index = data_dict["vocab"]["token_to_idx"]["<NULL>"]
    START_index = data_dict["vocab"]["token_to_idx"]["<START>"]
    END_index = data_dict["vocab"]["token_to_idx"]["<END>"]
    UNK_index = data_dict["vocab"]["token_to_idx"]["<UNK>"]

    return num_train, num_val

def main():
    num_train, num_val = get_data()
    
    rnn_model.eval()
    
    for split in ["train", "val"]:
        sample_idx = torch.randint(
            0, num_train if split == "train" else num_val, (VIS_BATCH_SIZE,)
        )
        sample_images = data_dict[split + "_images"][sample_idx]
        sample_captions = data_dict[split + "_captions"][sample_idx]
    
        # decode_captions is loaded from a5_helper.py
        gt_captions = decode_captions(sample_captions, data_dict["vocab"]["idx_to_token"])
    
        generated_captions = rnn_model.sample(sample_images.to(DEVICE))
        generated_captions = decode_captions(
            generated_captions, data_dict["vocab"]["idx_to_token"]
        )
    
        for i in range(VIS_BATCH_SIZE):
            plt.imshow(sample_images[i].permute(1, 2, 0))
            plt.axis("off")
            plt.title(
                f"[{split}] RNN Generated: {generated_captions[i]}\nGT: {gt_captions[i]}"
            )
            plt.show()

if __name__ == '__main__':
    main()

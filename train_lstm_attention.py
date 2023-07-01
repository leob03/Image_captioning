from rnn_lstm_captioning import AttentionLSTM, ImageEncoder, WordEmbedding, temporal_softmax_loss, CaptioningRNN

model = ImageEncoder(pretrained=True, verbose=True).to(device=DEVICE)

from a5_helper import train_captioner

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

    return num_train

def main():
    reset_seed(0)
    
    # data input
    small_num_train = get_data()
    sample_idx = torch.randint(num_train, size=(small_num_train,))
    small_image_data = data_dict["train_images"][sample_idx]
    small_caption_data = data_dict["train_captions"][sample_idx]
    
    # create the image captioning model
    attn_model = CaptioningRNN(
        cell_type="attn",
        word_to_idx=data_dict["vocab"]["token_to_idx"],
        input_dim=400,  # hard-coded, do not modify
        hidden_dim=512,
        wordvec_dim=256,
        ignore_index=NULL_index,
    )
    attn_model = attn_model.to(DEVICE)
    
    for learning_rate in [1e-3]:
        print("learning rate is: ", learning_rate)
        attn_model_submit, attn_loss_submit = train_captioner(
            attn_model,
            small_image_data,
            small_caption_data,
            num_epochs=60,
            batch_size=BATCH_SIZE,
            learning_rate=learning_rate,
            device=DEVICE,
        )

if __name__ == '__main__':
    main()

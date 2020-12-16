# All paths are relative to train_val.py file
config = {
	'images_path': '/usr/Flicker8k_Dataset/', #Make sure you put that last slash(/)
	'train_data_path': '/usr/Flickr_8k.trainImages.txt',
	'val_data_path': '/usr/Flickr_8k.devImages.txt',
	'captions_path': '/usr/Flickr8k.token.txt',
	'tokenizer_path': '/opt/imlabel1/imagedesc/model_data/tokenizer.pkl',
	'model_data_path': '/opt/imlabel1/imagedesc/model_data/', #Make sure you put that last slash(/)
	'model_load_path': '/opt/imlabel1/imagedesc/model_data/model_inceptionv3_epoch-20_train_loss-2.3481_val_loss-3.0575.hdf5',
	'num_of_epochs': 20,
	'max_length': 40, #This is set manually after training of model and required for test.py
	'batch_size': 64,
	'beam_search_k':7,
	'test_data_path': '/opt/twitterdata/images/', #Make sure you put that last slash(/)
	'model_type': 'inceptionv3', # inceptionv3 or vgg16
	'random_seed': 1035
}

rnnConfig = {
	'embedding_size': 300,
	'LSTM_units': 256,
	'dense_units': 256,
	'dropout': 0.3
}

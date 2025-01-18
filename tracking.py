from os import listdir
from xml.etree import ElementTree
from numpy import zeros
from numpy import asarray
from mrcnn.utils import Dataset
from matplotlib import pyplot
from mrcnn.visualize import display_instances
from mrcnn.utils import extract_bboxes
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from collections import Iterable
import tensorflow as tf
import mlflow
import mlflow.keras
import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
tf.compat.v1.disable_eager_execution()

# Original KangarooDataset class remains the same
class KangarooDataset(Dataset):
	# load the dataset definitions
	def load_dataset(self, dataset_dir, is_train=True):
		# define one class
		self.add_class("dataset", 1, "kangaroo")
		# define data locations
		images_dir = dataset_dir + '/images/'
		annotations_dir = dataset_dir + '/annots/'
		# find all images
		for filename in listdir(images_dir):
			# extract image id
			image_id = filename[:-4]
			# skip bad images
			if image_id in ['00090']:
				continue
			# skip all images after 150 if we are building the train set
			if is_train and int(image_id) >= 150:
				continue
			# skip all images before 150 if we are building the test/val set
			if not is_train and int(image_id) < 150:
				continue
			img_path = images_dir + filename
			ann_path = annotations_dir + image_id + '.xml'
			# add to dataset
			self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path)

	# extract bounding boxes from an annotation file
	def extract_boxes(self, filename):
		# load and parse the file
		tree = ElementTree.parse(filename)
		# get the root of the document
		root = tree.getroot()
		# extract each bounding box
		boxes = list()
		for box in root.findall('.//bndbox'):
			xmin = int(box.find('xmin').text)
			ymin = int(box.find('ymin').text)
			xmax = int(box.find('xmax').text)
			ymax = int(box.find('ymax').text)
			coors = [xmin, ymin, xmax, ymax]
			boxes.append(coors)
		# extract image dimensions
		width = int(root.find('.//size/width').text)
		height = int(root.find('.//size/height').text)
		return boxes, width, height

	# load the masks for an image
	def load_mask(self, image_id):
		# get details of image
		info = self.image_info[image_id]
		# define box file location
		path = info['annotation']
		# load XML
		boxes, w, h = self.extract_boxes(path)
		# create one array for all masks, each on a different channel
		masks = zeros([h, w, len(boxes)], dtype='uint8')
		# create masks
		class_ids = list()
		for i in range(len(boxes)):
			box = boxes[i]
			row_s, row_e = box[1], box[3]
			col_s, col_e = box[0], box[2]
			masks[row_s:row_e, col_s:col_e, i] = 1
			class_ids.append(self.class_names.index('kangaroo'))
		return masks, asarray(class_ids, dtype='int32')

	# load an image reference
	def image_reference(self, image_id):
		info = self.image_info[image_id]
		return info['path']

class KangarooConfig(Config):
    NAME = "kangaroo_cfg"
    NUM_CLASSES = 1 + 1
    STEPS_PER_EPOCH = 131

# Custom callback to log metrics to MLFlow
class MLFlowCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for metric_name, metric_value in logs.items():
            mlflow.log_metric(metric_name, metric_value, step=epoch)

def train_kangaroo_model(experiment_name="kangaroo-detection"):
    # Set up MLFlow experiment
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run():
        # Log parameters
        config = KangarooConfig()
        mlflow.log_params({
            "learning_rate": config.LEARNING_RATE,
            "epochs": 5,
            "steps_per_epoch": config.STEPS_PER_EPOCH,
            "batch_size": config.BATCH_SIZE,
            "num_classes": config.NUM_CLASSES
        })
        
        # Prepare datasets
        train_set = KangarooDataset()
        train_set.load_dataset('kangaroo', is_train=True)
        train_set.prepare()
        
        test_set = KangarooDataset()
        test_set.load_dataset('kangaroo', is_train=False)
        test_set.prepare()
        
        # Log dataset sizes
        mlflow.log_params({
            "train_size": len(train_set.image_ids),
            "test_size": len(test_set.image_ids)
        })
        
        # Define and configure the model
        model = MaskRCNN(mode='training', model_dir='./', config=config)
        
        # Load pre-trained weights
        model.load_weights('mask_rcnn_coco.h5', 
                         by_name=True, 
                         exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])
        
        # Create MLFlow callback
        mlflow_callback = MLFlowCallback()
        
        # Train the model
        history = model.train(train_set, 
                            test_set,
                            learning_rate=config.LEARNING_RATE,
                            epochs=5,
                            layers='heads',
                            custom_callbacks=[mlflow_callback])
        
        # Log the final model
        mlflow.keras.log_model(model.keras_model, "mask_rcnn_model")
        
        # Log training artifacts
        model_path = os.path.join(model.model_dir, f"{config.NAME}_final.h5")
        if os.path.exists(model_path):
            mlflow.log_artifact(model_path, "model_weights")
        
        # Log validation metrics
        final_val_loss = history.history.get('val_loss', [])[-1]
        mlflow.log_metric("final_val_loss", final_val_loss)
        
        return model, history

if __name__ == "__main__":
    # Run the training with MLFlow tracking
    model, history = train_kangaroo_model()
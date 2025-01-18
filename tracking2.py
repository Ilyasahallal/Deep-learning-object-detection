from os import listdir
from xml.etree import ElementTree
import numpy as np
from numpy import zeros, asarray, expand_dims
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mrcnn.config import Config
from mrcnn.model import MaskRCNN, mold_image
from mrcnn.utils import Dataset
import mlflow
import mlflow.keras
from sklearn.metrics import average_precision_score
import io
import os
import tempfile

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



class PredictionConfig(Config):
    NAME = "kangaroo_cfg"
    NUM_CLASSES = 1 + 1
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

def calculate_iou(box1, box2):
    """Calculate intersection over union between two boxes"""
    y1_1, x1_1, y2_1, x2_1 = box1
    y1_2, x1_2, y2_2, x2_2 = box2
    
    # Calculate intersection
    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_bottom = min(y2_1, y2_2)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # Calculate union
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - intersection_area
    
    return intersection_area / union_area

def evaluate_predictions(dataset, predictions, iou_threshold=0.5):
    """Calculate prediction metrics with error handling"""
    try:
        total_gt = 0
        total_pred = 0
        true_positives = 0
        
        # Make sure predictions list matches dataset length
        if len(predictions) > len(dataset.image_ids):
            predictions = predictions[:len(dataset.image_ids)]
        
        for i, image_id in enumerate(dataset.image_ids):
            if i >= len(predictions):
                break
                
            try:
                # Get ground truth boxes
                mask, _ = dataset.load_mask(i)
                gt_boxes = []
                for j in range(mask.shape[2]):
                    pos = np.where(mask[:,:,j])
                    if len(pos[0]) > 0:
                        y1, x1 = np.min(pos[0]), np.min(pos[1])
                        y2, x2 = np.max(pos[0]), np.max(pos[1])
                        gt_boxes.append([y1, x1, y2, x2])
                
                # Get predicted boxes
                pred_boxes = predictions[i]['rois']
                
                total_gt += len(gt_boxes)
                total_pred += len(pred_boxes)
                
                # Calculate matches using IoU
                for gt_box in gt_boxes:
                    for pred_box in pred_boxes:
                        if calculate_iou(gt_box, pred_box) >= iou_threshold:
                            true_positives += 1
                            break
                            
            except Exception as e:
                print(f"Warning: Error processing image {i}: {str(e)}")
                continue
        
        # Calculate metrics
        precision = true_positives / total_pred if total_pred > 0 else 0
        recall = true_positives / total_gt if total_gt > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'true_positives': true_positives,
            'total_predictions': total_pred,
            'total_ground_truth': total_gt
        }
        
        return metrics
        
    except Exception as e:
        print(f"Error in evaluate_predictions: {str(e)}")
        return {
            'precision': 0,
            'recall': 0,
            'f1_score': 0,
            'true_positives': 0,
            'total_predictions': 0,
            'total_ground_truth': 0
        }

def plot_actual_vs_predicted(dataset, model, cfg, n_images=5):
    """Plot and save comparison images with error handling"""
    predictions = []
    
    try:
        # Create a temporary directory for saving plots
        temp_dir = "temp_plots"
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        
        # Adjust n_images if dataset is smaller
        n_images = min(n_images, len(dataset.image_ids))
        
        plt.figure(figsize=(15, 5*n_images))
        
        for i in range(n_images):
            try:
                image = dataset.load_image(i)
                mask, _ = dataset.load_mask(i)
                scaled_image = mold_image(image, cfg)
                sample = expand_dims(scaled_image, 0)
                yhat = model.detect(sample, verbose=0)[0]
                predictions.append(yhat)
                
                # Plot actual
                plt.subplot(n_images, 2, i*2+1)
                plt.imshow(image)
                plt.title('Actual')
                for j in range(mask.shape[2]):
                    plt.imshow(mask[:, :, j], cmap='gray', alpha=0.3)
                
                # Plot predicted
                plt.subplot(n_images, 2, i*2+2)
                plt.imshow(image)
                plt.title('Predicted')
                ax = plt.gca()
                for box in yhat['rois']:
                    y1, x1, y2, x2 = box
                    width, height = x2 - x1, y2 - y1
                    rect = Rectangle((x1, y1), width, height, fill=False, color='red')
                    ax.add_patch(rect)
            
            except Exception as e:
                print(f"Warning: Error processing image {i}: {str(e)}")
                continue
        
        # Save plot to file
        plot_path = os.path.join(temp_dir, "predictions.png")
        plt.savefig(plot_path, format='png', bbox_inches='tight')
        plt.close()
        
        return predictions, plot_path
        
    except Exception as e:
        print(f"Error in plot_actual_vs_predicted: {str(e)}")
        return [], None

def run_prediction_with_mlflow(model_path, experiment_name="kangaroo-detection-inference"):
    try:
        mlflow.set_experiment(experiment_name)
        
        with mlflow.start_run() as run:
            print(f"MLFlow run ID: {run.info.run_id}")
            
            # Log model path and configuration
            cfg = PredictionConfig()
            mlflow.log_param("model_path", model_path)
            mlflow.log_params({
                "num_classes": cfg.NUM_CLASSES,
                "gpu_count": cfg.GPU_COUNT,
                "images_per_gpu": cfg.IMAGES_PER_GPU
            })
            
            # Load datasets
            train_set = KangarooDataset()
            train_set.load_dataset('kangaroo', is_train=True)
            train_set.prepare()
            
            test_set = KangarooDataset()
            test_set.load_dataset('kangaroo', is_train=False)
            test_set.prepare()
            
            # Log dataset sizes
            mlflow.log_params({
                "train_set_size": len(train_set.image_ids),
                "test_set_size": len(test_set.image_ids)
            })
            print("Dataset sizes logged")
            
            # Create and load model
            model = MaskRCNN(mode='inference', model_dir='./', config=cfg)
            model.load_weights(model_path, by_name=True)
            print("Model loaded")
            
            # Generate predictions and plots
            print("Generating train predictions...")
            train_predictions, train_plot_path = plot_actual_vs_predicted(train_set, model, cfg)
            if train_plot_path:
                mlflow.log_artifact(train_plot_path, "train_predictions.png")
            train_metrics = evaluate_predictions(train_set, train_predictions)
            print("Generating test predictions...")
            test_predictions, test_plot_path = plot_actual_vs_predicted(test_set, model, cfg)
            
            # Log plots if they were created successfully
            
            if test_plot_path:
                mlflow.log_artifact(test_plot_path, "test_predictions.png")
            
            # Calculate and log metrics
            print("Calculating metrics...")
            test_metrics = evaluate_predictions(test_set, test_predictions)
            
            # Log metrics with prefix
            print("Logging metrics...")
            for metric_name, value in train_metrics.items():
                print(f"train_{metric_name}: {value}")
                mlflow.log_metric(f"train_{metric_name}", float(value))
            
            for metric_name, value in test_metrics.items():
                print(f"test_{metric_name}: {value}")
                mlflow.log_metric(f"test_{metric_name}", float(value))
            
            # Clean up temporary files
            print("Cleaning up...")
            import shutil
            if os.path.exists("temp_plots"):
                shutil.rmtree("temp_plots")
            
            return train_metrics, test_metrics
            
    except Exception as e:
        print(f"Error in run_prediction_with_mlflow: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        print("Starting prediction script...")
        model_path = 'mask_rcnn_kangaroo_cfg_0001.h5'
        
        # Verify model file exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        train_metrics, test_metrics = run_prediction_with_mlflow(model_path)
        
        print("\nTraining Set Metrics:")
        for metric_name, value in train_metrics.items():
            print(f"{metric_name}: {value:.4f}")
        
        print("\nTest Set Metrics:")
        for metric_name, value in test_metrics.items():
            print(f"{metric_name}: {value:.4f}")
            
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nPlease ensure your files are organized correctly:")
        print("├── mask_rcnn_kangaroo_cfg_0005.h5")
        print("└── kangaroo/")
        print("    ├── images/")
        print("    │   ├── 00001.jpg")
        print("    │   └── ...")
        print("    └── annots/")
        print("        ├── 00001.xml")
        print("        └── ...")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
import os
import argparse
import json
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from transformers.image_utils import load_image
from tqdm import tqdm
import pickle
from collections import defaultdict

class CHAIREvaluator:
    """
    Implementation of CHAIR (Caption Hallucination Assessment with Image Relevance) metric
    Based on the paper "Object Hallucination in Image Captioning" by Rohrbach et al.
    """
    def __init__(self, coco_path, synonyms_path=None):
        """
        Initialize the CHAIR evaluator
        
        Args:
            coco_path: Path to the COCO dataset directory
            synonyms_path: Path to synonyms file (if None, will use default list)
        """
        self.coco_path = coco_path
        self.caption_file = os.path.join(coco_path, 'annotations/captions_val2014.json')
        self.instance_file = os.path.join(coco_path, 'annotations/instances_val2014.json')
        self.lemmatizer = WordNetLemmatizer()
        
        # Load COCO annotations
        print("Loading COCO annotations...")
        with open(self.caption_file, 'r') as f:
            self.captions = json.load(f)
        with open(self.instance_file, 'r') as f:
            self.instances = json.load(f)
        
        # Extract all COCO object categories
        self.coco_objects = [obj['name'] for obj in self.instances['categories']]
        
        # Process image to objects mapping
        self.imid_to_objects = self._get_imid_to_objects()
        
        # Load synonyms (could be provided or use default)
        self.synonyms = self._load_synonyms(synonyms_path)
        
        print(f"CHAIR evaluator initialized with {len(self.coco_objects)} object categories")
    
    def _load_synonyms(self, synonyms_path):
        """Load synonym mappings for COCO objects"""
        # Default synonyms if no file provided
        default_synonyms = {
            'person': ['child', 'girl', 'boy', 'man', 'woman', 'kid', 'baby', 'adult', 'rider', 'children', 'people', 'human'],
            'bicycle': ['bike'],
            'car': ['auto', 'automobile', 'vehicle'],
            'motorcycle': ['bike', 'motorbike'],
            'airplane': ['plane', 'jet', 'aircraft'],
            'bus': ['minibus'],
            'train': ['locomotive'],
            'truck': ['pickup'],
            'boat': ['ship', 'sailboat', 'canoe', 'vessel'],
            'traffic light': ['traffic signal', 'stop light'],
            'fire hydrant': ['hydrant'],
            'stop sign': ['traffic sign'],
            'parking meter': ['meter'],
            'bench': ['seat'],
            'bird': ['duck', 'goose', 'pigeon', 'fowl'],
            'cat': ['kitten', 'kitty', 'feline'],
            'dog': ['puppy', 'canine'],
            'horse': ['pony', 'stallion', 'equine'],
            'sheep': ['lamb'],
            'cow': ['bull', 'cattle', 'bovine'],
            'elephant': ['pachyderm'],
            'bear': ['grizzly', 'polar bear', 'panda'],
            'zebra': ['equid'],
            'giraffe': ['camelopard'],
            'backpack': ['knapsack', 'rucksack'],
            'umbrella': ['parasol'],
            'handbag': ['purse', 'bag'],
            'tie': ['necktie'],
            'suitcase': ['luggage', 'baggage'],
            'frisbee': ['disc'],
            'skis': ['ski'],
            'snowboard': ['snow board'],
            'sports ball': ['ball', 'baseball', 'football', 'soccer ball', 'basketball', 'volleyball'],
            'kite': ['wind kite'],
            'baseball bat': ['bat'],
            'baseball glove': ['glove'],
            'skateboard': ['board'],
            'surfboard': ['surfing board'],
            'tennis racket': ['racket'],
            'bottle': ['flask'],
            'wine glass': ['wine'],
            'cup': ['mug', 'glass', 'tea cup'],
            'fork': ['pitchfork'],
            'knife': ['blade', 'pocketknife'],
            'spoon': ['tablespoon', 'teaspoon'],
            'bowl': ['dish', 'basin'],
            'banana': ['plantain'],
            'apple': ['fruit'],
            'sandwich': ['burger', 'hamburger', 'cheeseburger'],
            'orange': ['citrus', 'fruit'],
            'broccoli': ['vegetable'],
            'carrot': ['vegetable'],
            'hot dog': ['frankfurter', 'sausage'],
            'pizza': ['pie'],
            'donut': ['doughnut', 'pastry'],
            'cake': ['pastry', 'dessert'],
            'chair': ['seat'],
            'couch': ['sofa', 'settee'],
            'potted plant': ['plant', 'houseplant'],
            'bed': ['mattress'],
            'dining table': ['table', 'desk'],
            'toilet': ['lavatory', 'restroom', 'bathroom'],
            'tv': ['television', 'monitor', 'screen'],
            'laptop': ['computer', 'notebook', 'pc'],
            'mouse': ['computer mouse'],
            'remote': ['remote control', 'controller'],
            'keyboard': ['keypad'],
            'cell phone': ['mobile phone', 'phone', 'smartphone'],
            'microwave': ['microwave oven'],
            'oven': ['stove'],
            'toaster': ['toast maker'],
            'sink': ['basin', 'washbasin'],
            'refrigerator': ['fridge'],
            'book': ['novel', 'textbook'],
            'clock': ['watch', 'timepiece'],
            'vase': ['jar', 'pot'],
            'scissors': ['shears'],
            'teddy bear': ['teddy', 'stuffed animal', 'plush'],
            'hair drier': ['hair dryer', 'blow dryer'],
            'toothbrush': ['tooth brush']
        }
        
        if synonyms_path and os.path.exists(synonyms_path):
            # Load from file if provided
            synonyms = {}
            with open(synonyms_path, 'r') as f:
                for line in f:
                    words = line.strip().split(',')
                    if words:
                        synonyms[words[0]] = words[1:]
            return synonyms
        else:
            return default_synonyms
    
    def _get_imid_to_objects(self):
        """Create a mapping from image ID to the objects present in that image"""
        imid_to_objects = defaultdict(set)
        
        for annotation in self.instances['annotations']:
            cat_id = annotation['category_id']
            # Get the object name for this category ID
            obj_name = next((cat['name'] for cat in self.instances['categories'] 
                            if cat['id'] == cat_id), None)
            if obj_name:
                imid_to_objects[annotation['image_id']].add(obj_name)
        
        return imid_to_objects
    
    def _normalize_word(self, word):
        """Lemmatize and lowercase a word"""
        return self.lemmatizer.lemmatize(word.lower())
    
    def _caption_to_words(self, caption):
        """
        Extract all content words from a caption and determine which are COCO objects
        
        Returns:
            words: All content words
            node_words: Words that correspond to COCO objects
            idxs: Indices of node_words in the original caption
            raw_words: Tokenized words before processing
        """
        # Tokenize
        raw_words = nltk.word_tokenize(caption.lower())
        
        # Extract content words (e.g., nouns) - could be more sophisticated with POS tagging
        words = [self._normalize_word(w) for w in raw_words if w.isalpha() and len(w) > 2]
        
        # Determine which words match COCO objects (directly or through synonyms)
        node_words = []
        idxs = []
        
        for idx, word in enumerate(words):
            if word in self.coco_objects:
                node_words.append(word)
                idxs.append(idx)
                continue
            
            # Check if word is a synonym of any COCO object
            for obj, syns in self.synonyms.items():
                if word in syns and obj in self.coco_objects:
                    node_words.append(obj)  # Add the canonical object name
                    idxs.append(idx)
                    break
        
        return words, node_words, idxs, raw_words
    
    def evaluate_caption(self, caption, image_id):
        """
        Evaluate a single caption for object hallucination
        
        Args:
            caption: The generated caption
            image_id: COCO image ID
            
        Returns:
            Dictionary with evaluation metrics and lists of hallucinated objects
        """
        # Get words and COCO objects in caption
        words, node_words, _, _ = self._caption_to_words(caption)
        
        # Get ground truth objects for this image
        gt_objects = self.imid_to_objects[image_id]
        
        # Find hallucinated objects (in caption but not in ground truth)
        hallucinated_words = [w for w in node_words if w not in gt_objects]
        
        # Calculate metrics
        metrics = {}
        if node_words:
            metrics['CHAIRi'] = len(hallucinated_words) / len(node_words)
        else:
            metrics['CHAIRi'] = 0.0
        
        metrics['CHAIRs'] = 1.0 if hallucinated_words else 0.0
        
        return {
            'image_id': image_id,
            'caption': caption,
            'mscoco_hallucinated_words': hallucinated_words,
            'mscoco_gt_words': list(gt_objects),
            'mscoco_generated_words': node_words,
            'metrics': metrics
        }
    
    def evaluate_captions(self, captions, image_ids):
        """
        Evaluate multiple captions for object hallucination
        
        Args:
            captions: List of captions
            image_ids: List of corresponding image IDs
            
        Returns:
            Dictionary with overall metrics and per-caption results
        """
        assert len(captions) == len(image_ids), "Number of captions must match number of image IDs"
        
        num_caps = len(captions)
        num_hallucinated_caps = 0
        hallucinated_word_count = 0
        coco_word_count = 0
        
        results = {'sentences': []}
        
        for i in tqdm(range(len(captions))):
            cap = captions[i]
            imid = image_ids[i]
            
            cap_dict = self.evaluate_caption(cap, imid)
            results['sentences'].append(cap_dict)
            
            # Update aggregate statistics
            if cap_dict['mscoco_hallucinated_words']:
                num_hallucinated_caps += 1
            
            hallucinated_word_count += len(cap_dict['mscoco_hallucinated_words'])
            coco_word_count += len(cap_dict['mscoco_generated_words'])
        
        # Calculate overall metrics
        chair_s = num_hallucinated_caps / num_caps if num_caps > 0 else 0
        chair_i = hallucinated_word_count / coco_word_count if coco_word_count > 0 else 0
        
        results['overall_metrics'] = {
            'CHAIRs': chair_s,
            'CHAIRi': chair_i
        }
        
        return results
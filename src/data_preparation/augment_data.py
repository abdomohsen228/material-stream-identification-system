# augment_data.py
import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import imgaug.augmenters as iaa
import matplotlib.pyplot as plt


class DataAugmenter:
    def __init__(self, raw_dir, out_dir, increase_by=0.4):
        # Resolve paths relative to project root
        script_dir = Path(__file__).parent.parent.parent  # Go up to project root
        self.raw_dir = (script_dir / raw_dir).resolve()
        self.out_dir = (script_dir / out_dir).resolve()
        self.increase_by = increase_by
        self.classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
        
        # make output dirs
        self.out_dir.mkdir(parents=True, exist_ok=True)
        for c in self.classes:
            (self.out_dir / c).mkdir(exist_ok=True)
    
    def get_augmentation_pipeline(self):
        # rotation, flips, scaling, color adjustments
        # Conservative settings to maintain accuracy with 40% increase
        aug = iaa.Sequential([
            iaa.Sometimes(0.65, iaa.Affine(rotate=(-40, 40), mode='reflect')),  # Slightly less rotation
            iaa.Fliplr(0.5),
            iaa.Flipud(0.3),
            iaa.Sometimes(0.5, iaa.Affine(scale={"x": (0.85, 1.15), "y": (0.85, 1.15)}, mode='reflect')),  # Tighter scaling range
            iaa.Sometimes(0.55, iaa.Sequential([
                iaa.Multiply((0.75, 1.25)),  # More conservative brightness
                iaa.LinearContrast((0.8, 1.4)),  # More conservative contrast
            ]))
        ])
        return aug
    
    def count_images(self):
        counts = {}
        total = 0
        print("Counting images...")
        for cls in self.classes:
            cls_dir = self.raw_dir / cls
            if cls_dir.exists():
                imgs = list(cls_dir.glob('*.jpg')) + list(cls_dir.glob('*.png'))
                counts[cls] = len(imgs)
                total += len(imgs)
                print(f"  {cls}: {len(imgs)}")
            else:
                counts[cls] = 0
                print(f"  {cls}: 0 (directory not found: {cls_dir})")
        
        print(f"Total: {total}\n")
        return counts, total
    
    def augment_images(self, class_name, n_to_generate, aug_pipeline):
        cls_dir = self.raw_dir / class_name
        out_dir = self.out_dir / class_name
        
        if not cls_dir.exists():
            return 0
        
        img_files = list(cls_dir.glob('*.jpg')) + list(cls_dir.glob('*.png'))
        if not img_files:
            print(f"No images found in {class_name}")
            return 0
        
        print(f"Augmenting {class_name}: generating {n_to_generate} images")
        
        count = 0
        idx = 0
        
        for i in tqdm(range(n_to_generate)):
            img_path = img_files[idx % len(img_files)]
            img = cv2.imread(str(img_path))
            
            if img is None:
                idx += 1
                continue
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            aug_img = aug_pipeline(image=img_rgb)
            aug_img = cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR)
            
            out_path = out_dir / f"{img_path.stem}_aug{i:04d}.jpg"
            cv2.imwrite(str(out_path), aug_img)
            
            count += 1
            idx += 1
        
        return count
    
    def run(self):
        print("Data Augmentation Starting...")
        print("-" * 50)
        
        counts, total = self.count_images()
        
        # figure out how many to make per class
        target_total = int(total * (1 + self.increase_by))
        to_generate = target_total - total
        
        print(f"Target increase: {self.increase_by*100:.0f}%")
        print(f"Will generate ~{to_generate} new images\n")
        
        aug_pipeline = self.get_augmentation_pipeline()
        
        # distribute roughly proportionally, ensuring minimum 40% per class
        for cls in self.classes:
            if counts[cls] > 0:
                min_per_class = int(counts[cls] * self.increase_by)  # At least 40% per class
                proportion = counts[cls] / total
                n_aug = max(min_per_class, int(to_generate * proportion))
                
                if n_aug > 0:
                    self.augment_images(cls, n_aug, aug_pipeline)
        
    
    def show_examples(self, class_name='cardboard', n=5):
        cls_dir = self.raw_dir / class_name
        imgs = list(cls_dir.glob('*.jpg')) + list(cls_dir.glob('*.png'))
        
        if not imgs:
            print(f"No images found in {class_name}")
            return
        
        img = cv2.imread(str(imgs[0]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        aug = self.get_augmentation_pipeline()
        
        fig, axes = plt.subplots(1, n+1, figsize=(15, 3))
        axes[0].imshow(img)
        axes[0].set_title('Original')
        axes[0].axis('off')
        
        for i in range(n):
            aug_img = aug(image=img)
            axes[i+1].imshow(aug_img)
            axes[i+1].set_title(f'Aug {i+1}')
            axes[i+1].axis('off')
        
        plt.tight_layout()
        # Save to reports directory relative to project root
        script_dir = Path(__file__).parent.parent.parent
        reports_dir = script_dir / 'reports'
        reports_dir.mkdir(exist_ok=True)
        save_path = reports_dir / 'augmentation_examples.png'
        plt.savefig(save_path, dpi=150)
        print(f"Saved examples to {save_path}")


if __name__ == '__main__':
    # Get project root (parent of src)
    project_root = Path(__file__).parent.parent.parent
    reports_dir = project_root / 'reports'
    reports_dir.mkdir(exist_ok=True)
    
    augmenter = DataAugmenter('data/raw', 'data/augmented', increase_by=0.40)
    augmenter.run()
    augmenter.show_examples()

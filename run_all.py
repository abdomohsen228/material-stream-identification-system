import sys
import subprocess
from pathlib import Path
from datetime import datetime
def print_header(step_num, step_name):
    print("\n" + "=" * 70)
    print(f"STEP {step_num}: {step_name}")
    print("=" * 70 + "\n")
def run_step(step_num, step_name, script_path, description):
    print_header(step_num, step_name)
    print(f"Description: {description}")
    print(f"Running: {script_path}\n")
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            check=True,
            capture_output=False,
            text=True
        )
        print(f"\n Step {step_num} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n Step {step_num} failed with error code {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"\n Script not found: {script_path}")
        return False
    except Exception as e:
        print(f"\n Error running step {step_num}: {e}")
        return False
def check_prerequisites():
    print("Checking prerequisites...")
    root = Path(__file__).parent
    checks = [
        (root / "src" / "data_preparation" / "augment_data.py", "Augmentation script"),
        (root / "data" / "raw", "Raw data directory"),
    ]
    all_ok = True
    for path, name in checks:
        if path.exists():
            print(f"   {name}: {path}")
        else:
            print(f"   {name} NOT FOUND: {path}")
            all_ok = False
    
    return all_ok

def main():
    root = Path(__file__).parent
    start_time = datetime.now()
    
    print("\n" + "=" * 70)
    print("MATERIAL STREAM IDENTIFICATION SYSTEM - COMPLETE PIPELINE")
    print("=" * 70)
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    if not check_prerequisites():
        print("\nSome prerequisites are missing. Please check the paths above.")
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Pipeline aborted.")
            return
    steps = [
        {
            "num": 1,
            "name": "Data Augmentation",
            "script": root / "src" / "data_preparation" / "augment_data.py",
            "description": "Generate augmented images (40% increase) with conservative settings"
        },
        {
            "num": 2,
            "name": "Feature Extraction",
            "script": root / "src" / "feature_extraction" / "extract_features.py",
            "description": "Extract CNN features, HOG, LBP, Haralick, and color histograms"
        },
        {
            "num": 3,
            "name": "Model Training",
            "script": root / "src" / "models" / "svm.py",
            "description": "Train SVM classifier with hyperparameter tuning"
        },
        {
            "num": 4,
            "name": "Test Rejection Mechanism",
            "script": root / "src" / "models" / "test_rejection_mechanism.py",
            "description": "Test unknown class rejection mechanism",
            "optional": True
        },
    ]
    results = []
    for step in steps:
        success = run_step(
            step["num"],
            step["name"],
            step["script"],
            step["description"]
        )
        results.append((step["name"], success))
        if not success and not step.get("optional", False):
            print(f"\nCritical step failed. Pipeline stopped.")
            print("You can continue manually or fix the issue and re-run.")
            break 
        if step["num"] < len(steps):
            if step.get("optional", False):
                continue
            print("\n" + "-" * 70)
            response = input("Press Enter to continue to next step (or 'q' to quit): ")
            if response.lower() == 'q':
                print("\nPipeline stopped by user.")
                break
    end_time = datetime.now()
    duration = end_time - start_time
    print("\n" + "=" * 70)
    print("PIPELINE SUMMARY")
    print("=" * 70)
    for step_name, success in results:
        status = "SUCCESS" if success else "FAILED"
        print(f"{step_name:30s} : {status}")
    print(f"\nTotal duration: {duration}")
    print(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n" + "=" * 70)
    print("ADDITIONAL TESTING OPTIONS")
    print("=" * 70)
    print("\nTo test predictions with rejection mechanism:")
    print("  1. Single images:    python test_unknown_images.py")
    print("  2. Batch images:     python test_batch_imgs.py")
    print("\n" + "=" * 70)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        sys.exit(1)


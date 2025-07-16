import os
import sys
from huggingface_hub import snapshot_download

def download_model_from_mirror(model_name: str, mirror_url: str, save_directory: str):
    """
    Downloads a Hugging Face model from a specified mirror URL to a local directory.

    Args:
        model_name (str): The name of the model on Hugging Face Hub (e.g., "bert-base-uncased", "gpt2").
        mirror_url (str): The base URL of the Hugging Face mirror (e.g., "https://hf-mirror.com").
        save_directory (str): The local path where the model files will be saved.
    """
    # Store the original HF_ENDPOINT value if it exists
    original_hf_endpoint = os.environ.get('HF_ENDPOINT')

    try:
        print(f"Setting HF_ENDPOINT to: {mirror_url}")
        # Set the environment variable to point to the mirror
        os.environ['HF_ENDPOINT'] = mirror_url

        print(f"Starting download of model '{model_name}' from mirror...")
        print(f"Saving to: {save_directory}")

        # Use snapshot_download to download all files from the model repository snapshot
        # local_dir specifies the target directory
        # local_dir_use_symlinks=False ensures files are copied directly, not symlinked
        download_path = snapshot_download(
            repo_id=model_name,
            local_dir=save_directory,
            local_dir_use_symlinks=False,
            # Add allow_patterns/ignore_patterns if you only need specific files
            # allow_patterns=["*.bin", "*.json", "*.txt"]
        )

        print(f"Download complete! Model files saved to: {download_path}")

    except Exception as e:
        print(f"An error occurred during the download: {e}", file=sys.stderr)
        # It's good practice to print errors to stderr

    finally:
        # Restore the original HF_ENDPOINT environment variable
        if original_hf_endpoint is not None:
            print(f"Restoring original HF_ENDPOINT: {original_hf_endpoint}")
            os.environ['HF_ENDPOINT'] = original_hf_endpoint
        else:
            # If original was not set, remove the one we set
            if 'HF_ENDPOINT' in os.environ:
                print("Unsetting HF_ENDPOINT")
                del os.environ['HF_ENDPOINT']

# --- Example Usage ---
if __name__ == "__main__":
    # --- Configuration ---
    # 替换为你想要下载的模型名称
    model_to_download = "BAAI/bge-m3"

    # 替换为你使用的镜像站地址
    # 例如: "https://hf-mirror.com" 或 其他可用镜像站
    mirror_site_url = "https://hf-mirror.com"
    #mirror_site_url = None

    # 替换为你想要保存模型的本地路径
    # 如果目录不存在，脚本会尝试创建
    save_directory_path = f"C:/Users/94373/Desktop/RAG/models/{model_to_download}" # Example path

    # --- Run the download function ---
    print("-" * 30)
    print("Hugging Face Model Mirror Downloader")
    print("-" * 30)
    print(f"Model: {model_to_download}")
    print(f"Mirror URL: {mirror_site_url}")
    print(f"Save Directory: {save_directory_path}")
    print("-" * 30)

    download_model_from_mirror(model_to_download, mirror_site_url, save_directory_path)

    print("-" * 30)
    print("Script finished.")
    print("-" * 30)
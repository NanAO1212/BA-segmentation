#!/usr/bin/env python3
import subprocess
import sys

def quick_test():
    """本地快速测试"""
    print("Running quick test with sample data...")
    subprocess.run(["python", "scripts/train.py", "--debug", "--epochs", "2"])

def sync_to_github():
    """同步到GitHub"""
    print("Pushing to GitHub...")
    subprocess.run(["git", "add", "."])
    commit_msg = input("Commit message: ")
    subprocess.run(["git", "commit", "-m", commit_msg])
    subprocess.run(["git", "push", "origin", "main"])

def prepare_for_autodl():
    """准备AutoDL训练"""
    print("1. Make sure your code is tested locally")
    print("2. Push to GitHub: python dev_tools.py --sync")
    print("3. SSH to AutoDL and run: bash autodl_setup.sh")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "--test":
            quick_test()
        elif sys.argv[1] == "--sync":
            sync_to_github()
        elif sys.argv[1] == "--prepare":
            prepare_for_autodl()

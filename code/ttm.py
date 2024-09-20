import git
import shutil
import os
def download_and_install():
    # URL of the repository you want to clone
    repo_url = 'https://github.com/IBM/tsfm.git'
    # Local directory where the repository will be cloned
    local_dir = '../ttm/'

    # Check if the directory exists and remove it if it does
    if os.path.exists(local_dir):
        shutil.rmtree(local_dir)

    # Clone the repository
    git.Repo.clone_from(repo_url, local_dir)

    # Remove the .git directory
    git_dir = os.path.join(local_dir, '.git')
    if os.path.exists(git_dir):
        shutil.rmtree(git_dir)

download_and_install()
import pandas as pd
from sklearn.decomposition import PCA

def create_col_names(no_of_components):
    x = 1
    columns = []
    while x <= no_of_components:
        columns.append(f'PC{x}')
        x += 1
    return columns

def filter_PCAs_via_eigen(principal_components, pca):
    eigenvalues = pca.explained_variance_
    # Filter PCs with eigenvalues greater than 1
    pcs_greater_than_1_indices = [i for i, eigenvalue in enumerate(eigenvalues) if eigenvalue > 1]
    print("Principal Components with Eigenvalues > 1:", pcs_greater_than_1_indices)
    filtered_principal_components = principal_components[:, pcs_greater_than_1_indices]
    return pcs_greater_than_1_indices, filtered_principal_components

def pca_feature_extraction(X_scaled):

    pca = PCA()  # Choose the number of components
    principal_components = pca.fit_transform(X_scaled)
    # keep only PCs with eigen value more than 1
    pcs_greater_than_1_indices, filtered_principal_components = filter_PCAs_via_eigen(principal_components, pca)
    columns = create_col_names(len(pcs_greater_than_1_indices))
    pca_df = pd.DataFrame(data=filtered_principal_components, columns=columns)
    
    return pca_df
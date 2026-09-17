import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt

from .symnmf import SymNMF

def assignModules(W, gene_names, name, output_folder, **kwargs):
    """
    Assign modules using specified pleiotropy function. Outputs assignments and diagnostics.
    Parameters:
        W: DataFrame (genes x modules, no gene names)
        gene_names: list of gene names in order of W
        func: pleiotropy assignment function
        name: string to save module assignments
    Returns:
        module_assignments csv: (Gene, Module)
        module_assignments_grouped csv: (Gene, Module) grouped by module 
    """
   

    module_assignments = cumulative_gene_mass(W = W, gene_names=gene_names, output_folder=output_folder, name=name, **kwargs)
    module_assignments.to_csv(f"{output_folder}/module_assignments_{name}.csv", index=False)

    module_assignments_grouped = module_assignments.copy()
    module_assignments_grouped = module_assignments_grouped.sort_values("Module", ascending=[False])
    module_assignments_grouped.to_csv(f"{output_folder}/module_assignments_grouped_{name}.csv", index=False)



    return module_assignments


#this function runs symmetric NMF
def runSymNMF(A, module_number, name, output_folder, gene_names, save=True, init_matrix = None):

    '''
    parameters: 
        A -> local correlation matrix on which nmf should run
        module_number -> number of desired modules
        init_matrix -> matrix to initialize symNMF with 
        name -> string to save the loading matrix with

    outputs:
        W -> matrix of loadings (dataframe)
    
    '''
    if init_matrix is None:
    
        alpha = 10
        np.random.seed(50)
        k = module_number
        size = len(gene_names)
        init_matrix = np.random.dirichlet(alpha * np.ones(k), size)

    snmf = SymNMF(module_number)
    snmf.fit(A, init_matrix)

    #save the matrix of loadings as a dataframe
    W = pd.DataFrame(snmf.W, index=gene_names)
    if save:
        W.to_csv(f"{output_folder}/{name}_loadings_matrix.csv", index=True)
    
    print("Ran symNMF\n")
    return W


def cumulative_gene_mass(W, alpha, gene_names, output_folder, name):

    """ saves: pleiotropy scores csv: list of genes and associated pleiotropy scores (descending order)"""

    if alpha == 0:
        module_assignments = row_max(W)
        return module_assignments

    H = W.copy()
    
    #take W, and for each row, divide each element in the row by its row sum
    H_sq = H ** 2
    
    # H_sq is a DataFrame indexed by gene names
    scores = H_sq.apply(lambda row: pleiotropy_score(row.to_numpy()), axis=1)

    # 0–1 min-max normalize (order preserved) + round to 4 decimals
    smin, smax = scores.min(), scores.max()
    scores_norm = (scores - smin) / (smax - smin) 

    pleiotropy_scores = scores_norm.round(4).reset_index()
    pleiotropy_scores.columns = ["Gene", "Score"]
    pleiotropy_scores = pleiotropy_scores.sort_values(by="Score", ascending=False)



    pleiotropy_scores.to_csv(f"{output_folder}/pleiotropy_scores_{name}.csv", index=False)


    #normalize with sum of squares 
    
    row_sums = H_sq.sum(axis=1)
    H_norm = H_sq.div(row_sums, axis=0)
    

    #sort each row and take cumulative sum, and keep the top threshold % of each row's cumulative mass
   
    for idx in H_norm.index:
        sorted_vals = H_norm.loc[idx].sort_values(ascending=False)
        cumsum = sorted_vals.cumsum()
        total = cumsum.iloc[-1]
        cutoff_n = (cumsum >= alpha * total).values.searchsorted(True) + 1
        cutoff_idx = sorted_vals.index[:cutoff_n]

        # Set everything below cutoff to 0
        H_norm.loc[idx, ~H_norm.columns.isin(cutoff_idx)] = 0

   

    #convert H_norm to a binary matrix, and multiply by W to get H_thresh
    mask = (H_norm > 0).astype(float)
    H_thresh = W * mask  # Retains original values where thresholded, zeros elsewhere


    #use the kept elements to make module assignments 
    assignments = []

    for gene_idx, gene in enumerate(gene_names):
        for module_idx in range(H_thresh.shape[1]):
            if H_thresh.iloc[gene_idx, module_idx] > 0:
                assignments.append({
                    "Gene": gene,
                    "Module": module_idx + 1  # shift module index to start from 1
                })

    module_assignments = pd.DataFrame(assignments)
    return module_assignments

def pleiotropy_score(x):
    nz = x[x!= 0]
    var = float(np.var(nz, ddof=0))
    return float(nz.size) / (var + 1.0)



def runPCA(adata, module_assignments, variance_threshold=0.70):
    """
    Run PCA for each module subset of genes, keep enough PCs to explain 
    `variance_threshold` of variance, and output concatenated PCs.
    """
    #print(module_assignments.shape)
    pcs_by_module = []
    pc_loading_list = []
    
    for i in sorted(module_assignments['Module'].unique()):
        genes_k = module_assignments.loc[module_assignments['Module'] == i, "Gene"].drop_duplicates()
        adata_k = adata[:, adata.var_names.isin(genes_k)].copy()
        #print(f"Module {i} adata shape: ", adata_k.shape)
        
        # Run PCA with full rank
        sc.tl.pca(adata_k, svd_solver='arpack', zero_center=True)
        
        # Compute cumulative explained variance
        explained = adata_k.uns['pca']['variance_ratio']
        cumsum = np.cumsum(explained)
        n_comps = np.searchsorted(cumsum, variance_threshold) + 1


        #print(f"Keeping {n_comps} components for module {i} to reach {variance_threshold*100:.1f}% variance")
        
        pcs_k = adata_k.obsm["X_pca"][:, :n_comps].astype(np.float32, copy=False)
        pcs_by_module.append(pcs_k)

        #save loadings for pcs

        Lk = adata_k.varm["PCs"][:, :n_comps].astype(np.float32, copy=False)
        df = pd.DataFrame(
            Lk,
            index=adata_k.var_names,
            columns=[f"M{i}_PC{j+1}" for j in range(n_comps)]
        ).reindex(adata.var_names, fill_value=0.0)

        pc_loading_list.append(df)




    full_pcs = np.ascontiguousarray(np.hstack(pcs_by_module), dtype=np.float32)
    pc_loadings_df = pd.concat(pc_loading_list, axis=1)

    pc_names = list(pc_loadings_df.columns)

    #print("Ran PCA\n ")
    #print("final shape of cell x pcs matrix: ", full_pcs.shape)



    return full_pcs, pc_loadings_df, pc_names
        

def umap(name, adata, module_assignments, color, adata_path, umap):

    pcs, loadings, pc_names = runPCA(adata, module_assignments, variance_threshold=0.70)

    adata.obsm[f"X_pcs_{name}"] = pcs 
    adata.uns[f"X_pc_names_{name}"] = pc_names
    adata.varm[f"PCs_{name}"] = loadings.to_numpy(dtype=np.float32) #genes x concatenated modpcs 

    adata.write(adata_path)
    print("pcs saved to anndata")

    if umap:
    
        # Compute neighbors based on these PCs
        sc.pp.neighbors(adata, use_rep=f"X_pcs_{name}", n_neighbors=30, key_added=f"neighbors_{name}")
    
        # Run UMAP embedding using those neighbors
    
        sc.tl.umap(adata, neighbors_key=f"neighbors_{name}")
        adata.obsm[f"X_umap_{name}"] = adata.obsm["X_umap"].copy()

        sc.pl.embedding(adata, basis=f"X_umap_{name}", color=color, title=f"{name}_UMAP", palette=sc.pl.palettes.default_102, save=f"_umap")
    

        print("Done with umap!")



def plot_module_distribution(module_assignments):
    # Load CSV
       # Load CSV
    df = module_assignments 

    # Count # of modules per gene
    module_counts = df.groupby("Gene")["Module"].nunique()

    # Count how many genes fall into each category (#modules = 1, 2, 3, ...)
    count_dist = module_counts.value_counts().sort_index()

    # Plot
    plt.figure(figsize=(6, 4), dpi=150)
    bars = plt.bar(count_dist.index, count_dist.values, edgecolor="black")

    # Label each bar with the count
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            str(int(height)),
            ha='center',
            va='bottom',
            fontsize=10
        )

    # Labels and title
    plt.xlabel("Number of Modules Assigned to Gene")
    plt.ylabel("Number of Genes")
    plt.title(f"Distribution of Module Assignments")
    plt.xticks(count_dist.index)
    plt.tight_layout()
    plt.show()


#this function will assign modules based on the maximum of each row (gene) in the loading matrix (hard clustering)
def row_max(W):
    W_array = W.values
    assignments = np.argmax(W_array, axis=1) + 1 #so that module assignments start at 1

    module_assignments =  pd.DataFrame({
        "Gene": W.index,
        "Module": assignments
    })


    return module_assignments  
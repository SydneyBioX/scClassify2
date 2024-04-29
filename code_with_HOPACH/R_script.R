## You need to set your own R path
.libPaths('./Your Path')
library(BiocParallel)
library(SingleCellExperiment)
library(Seurat)
library(hopach)
library(scClassify)
library(reticulate)

# You need to set your own train_data path, should be a .rds file
train_data=readRDS("./train_data.rds")
hvgs=getTopHVGs(train_data, n=600)
train_data=train_data[hvgs]

# You need to set your own python environment path
use_python("/miniconda3/envs/scclass/bin/python")  

adata <- import("anndata")
scipy <- import("scipy.sparse")
py_annData <- adata$AnnData(X = scipy$csr_matrix(t(as.matrix(counts(train_data)))), obs = as.data.frame(train_data$CellType), var = as.data.frame(rowData(train_data)))
py_annData$write_h5ad("./train_data.h5ad")


table(train_data$CellType)
unique_cell_types <- unique(train_data$CellType)
average_expression_by_cell_type <- lapply(unique_cell_types, function(ct) {
  subset_data <- counts(train_data)[, train_data$CellType == ct]
  row_means <- rowMeans(subset_data)
  
  # Return the result with cell type name as the list name
  return(list(cell_type = ct, average_expression = row_means))
})
average_expression_matrices <- average_expression_by_cell_type[[1]][[2]]
celltypes=list()
for (i in 1:10){
  celltypes[[i]]=average_expression_by_cell_type[[i]][[1]]
  average_expression_matrices <-cbind(average_expression_matrices, average_expression_by_cell_type[[i]][[2]])
}
average_expression_matrices=average_expression_matrices[,-1]
res <- runHOPACH(data =t(average_expression_matrices))

tree_using<-res$cutree_list[[2]]

for(i in 1:length(tree_using)) {
  line_to_write <- paste(c(as.character(celltypes[[i]]), ":", tree_using[[i]]), collapse = "")
  write(line_to_write, "./tree.txt", append = TRUE)
  print(as.character(celltypes[[i]]))
  print(tree_using[[i]])
}

## Must change few paths first and then run this line for the training 
system('python main.py')
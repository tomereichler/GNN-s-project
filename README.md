# GAT project

Data Analysis and Prediction on Academic Citation Networks: 
The main purpose of this analysis is to predict the categories of academic articles. By using the information contained in the citation network and the associated feature vectors, we can develop a predictive model that accurately assigns categories to articles automatically. This model will help organize and facilitate researchers' access to relevant articles in their field of interest.

After several experiments with different models, we determined that Graph Attention Network (GAT) is the most appropriate choice for this task. The GAT model uses the Attention layer to efficiently capture the relationships between articles in the network.
This model consists of two types of layers: GatConv and Gatv2Conv. In our implementation, we chose to use Gatv2Conv layers due to their improved performance, which is achieved by changing the order of operations within the model architecture. Specifically, the weight matrix W is applied after the concatenation operation, while the attention matrix is applied after the LeakyReLU activation function.

Gatv2Conv layers in the GAT model allow us to extract and integrate structural and semantic information in the network. By considering the citation relationships and aggregated feature vectors, the model can learn to identify patterns and dependencies indicative of article categories. The attention mechanism within GAT allows the model to focus on the most relevant articles during the prediction process, which improves its prediction accuracy.

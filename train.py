import torch
import pandas as pd

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from data_load_movies import load_movies
from data_load_ratings import load_ratings
import argparse
import os
import random
from pathlib import Path
from tqdm import tqdm


import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader, LinkNeighborLoader
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.transforms import RandomLinkSplit, ToUndirected
from sklearn.preprocessing import LabelEncoder, StandardScaler

from model.bipartite_sage import MetaSage
from model.bipartite_gat import MetaGATv2
from model.model import Model
from model.graph_transformer import MetaTransformerGat

def load_data(args):
    movies, movie_mappings = load_movies(args.word_embeddings)


    ratings, user_mappings = load_ratings()


    src = [user_mappings[index] for index in ratings.index]
    dst = [movie_mappings[index] for index in ratings['movieId']]
    edge_index = torch.tensor([src, dst])
    edge_attrs = [
        torch.tensor(ratings[column].values).unsqueeze(dim=1) for column in ['rating', 'timestamp'] 
    ]
    edge_label = torch.cat(edge_attrs, dim=-1)

    # movies = movies.fillna(0).astype(np.float32)
    products_tensor = torch.from_numpy(movies.values).to(torch.float32) # TODO: PCA
    # customer_tensor = torch.from_numpy(ratings.index.values).to(torch.float32).unsqueeze(dim=-1)
    # customer_tensor = torch.arange(len(user_mappings)).to(torch.float32).unsqueeze(dim=-1)
    # customer_tensor = torch.rand(len(user_mappings), 1)
    customer_tensor = torch.arange(len(user_mappings))
    """
    TODO:
    Feature Engineering: If user IDs are all you have, consider engineering synthetic features. This could include:
    Node degree (number of connections each user has with various products).
    Clustering coefficient (a measure of the degree to which nodes in a graph tend to cluster together).
    Any available metadata from interaction patterns (e.g., frequency or recency of user-product interactions).
    """

    # print(products_tensor.shape)
    # print(customer_tensor.shape)
    # print(customer_tensor)

    data = HeteroData()
    # data['customer'].num_nodes = len(user_mappings)
    data['customer'].x = customer_tensor
    data['product'].x = products_tensor

    data['customer', 'buys', 'product'].edge_index = edge_index
    data['customer', 'buys', 'product'].edge_label = edge_label


    data = ToUndirected()(data)

    train_data, val_data, test_data = RandomLinkSplit(
        num_val=args.val_split,
        num_test=args.test_split,
        is_undirected=True,
        neg_sampling_ratio=0.0,
        edge_types=[('customer', 'buys', 'product')],
        rev_edge_types=[('product', 'rev_buys', 'customer')],
    )(data)

    if args.model in ["meta_sage", "meta_gatv2"]:
        # Generate the co-occurence matrix of movies<>movies:
        metapath = [('product', 'rev_buys', 'customer'), ('customer', 'buys', 'product')]
        train_data = T.AddMetaPaths(metapaths=[metapath])(train_data)

        # Apply normalization to filter the metapath:
        _, edge_weight = gcn_norm(
            train_data['product', 'product'].edge_index,
            num_nodes=train_data['product'].num_nodes,
            add_self_loops=False,
        )
        edge_index = train_data['product', 'product'].edge_index[:, edge_weight > 0.002]
        # train_data['product', 'metapath_0', 'product'].edge_index = edge_index

        # TODO: Metapaths for customers. Doesnt make much sense if there are no features for customers

        train_data['product', 'metapath_0', 'product'].edge_index = edge_index
        val_data['product', 'metapath_0', 'product'].edge_index = edge_index
        test_data['product', 'metapath_0', 'product'].edge_index = edge_index

    # print(train_data)
    return train_data, val_data, test_data

def split_data(data, val_ratio=0.15, test_ratio=0.15):
    transform = RandomLinkSplit(
        num_val=val_ratio,
        num_test=test_ratio,
        is_undirected=True,
        neg_sampling_ratio=0.0,
        edge_types=[('customer', 'buys', 'product')],
        rev_edge_types=[('product', 'rev_buys', 'customer')],
    )
    return transform(data)

def weighted_mse_loss(pred, target, weight=None):
    weight = torch.tensor([1.]) if weight is None else weight[target.to('cpu').long()].to(pred.dtype)
    # diff = pred - target.to(pred.dtype)
    # weighted_diff = weight * diff.pow(2)
    # sum_loss = weighted_diff
    # loss = sum_loss.mean()
    loss = (weight.to(pred.device) * (pred - target.to(pred.dtype)).pow(2)).mean()
    return loss

def train(model, data_loader, optimizer, weight=None, scheduler=None, args=None, wandb=None):
    model.train()
    total_loss = total_nodes = 0
    i = 0
    for data in tqdm(data_loader):
        data.to(args.device)
        optimizer.zero_grad()
        pred = model(data.x_dict, data.edge_index_dict,
                     data['customer', 'product'].edge_label_index,
                     edge_label=data['product', 'rev_buys', 'customer'].edge_label[:,1:]
                     ).squeeze(axis=-1)
        target = data['customer', 'product'].edge_label[:,0]
        # loss = weighted_mse_loss(pred, target, weight) # TODO: Add some other loss
        loss = F.mse_loss(pred, target.to(pred.dtype))
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()
        if i % 5 == 0 and scheduler:
            wandb.log({"train_lr": scheduler.get_last_lr()[0]},
                      # commit=False, # Commit=False just accumulates data
                      )

        total_loss += loss.item() * pred.numel()
        total_nodes += pred.numel()

    return float(total_loss/total_nodes)

import torch  # Ensure torch is imported

@torch.no_grad()
def old_test(model, data):
    model.eval()
    pred = model(data.x_dict, data.edge_index_dict,
                 data['customer', 'product'].edge_label_index,
                 edge_label=data['product', 'rev_buys', 'customer'].edge_label[:,1:]
                 ).squeeze(axis=-1)
    pred = pred.clamp(min=0, max=5)
    target = data['customer', 'product'].edge_label[:,0].float()
    rmse = F.mse_loss(pred, target).sqrt()
    return float(rmse)

@torch.no_grad()
def test(model, loader):
    model.eval()  # Ensure the model is in evaluation mode
    total_sq_error = 0  # Total squared error
    total_samples = 0   # Total number of samples

    for data in loader:
        pred = model(data.x_dict, data.edge_index_dict,
                     data['customer', 'product'].edge_label_index,
                     edge_label=data['product', 'rev_buys', 'customer'].edge_label[:,1:]
                     ).squeeze(-1)
        pred = pred.clamp(min=0, max=5)  # Clamping predictions
        target = data['customer', 'product'].edge_label[:,0].float()
        
        # Calculate squared error and accumulate
        target = target.to(pred.device)
        sq_error = F.mse_loss(pred, target, reduction='sum')
        total_sq_error += sq_error.item()
        total_samples += data.num_nodes  # Update the count of total samples

    # Calculate the average of the squared errors, then take the square root using torch.sqrt
    average_rmse = torch.sqrt(torch.tensor(total_sq_error / total_samples))
    return average_rmse




# @torch.no_grad()
# def test(model, data):
#     pred = model(data.x_dict, data.edge_index_dict,
#                  data['customer', 'product'].edge_label_index,
#                  edge_label=data['product', 'rev_buys', 'customer'].edge_label[:,1:]
#                  ).squeeze(axis=-1)
#     pred = pred.clamp(min=0, max=5)
#     target = data['customer', 'product'].edge_label[:,0].float()
#     rmse = F.mse_loss(pred, target).sqrt()
#     return float(rmse)


@torch.no_grad()
def top_at_k(model, src, dst, train_data, test_data, k=10):
    customer_idx = random.randint(0, len(src) - 1)
    customer_row = torch.tensor([customer_idx] * len(dst))
    all_product_ids = torch.arange(len(dst))
    edge_label_index = torch.stack([customer_row, all_product_ids], dim=0)
    pred = model(train_data.x_dict, train_data.edge_index_dict,
                 edge_label_index)
    pred = pred.clamp(min=0, max=5)

    # we will only select movies for the user where the predicting rating is =5
    rec_product_ids = (pred[:, 0] == 5).nonzero(as_tuple=True)
    top_k_recommendations = [rec_product for rec_product in rec_product_ids[0].tolist()[:k]]

    test_edge_label_index = test_data['customer', 'product'].edge_label_index
    customer_interacted_products = test_edge_label_index[1, test_edge_label_index[0] == customer_idx]

    hits = 0
    for product_idx in top_k_recommendations:
        if product_idx in customer_interacted_products: hits += 1

    return hits / k

def find_first_component(variance_ratio, theshold=0.9):
    cumulative_variance = np.cumsum(variance_ratio)
    for i, ratio in enumerate(cumulative_variance):
        if ratio > theshold:
            return i + 1
    return -1

def plot_variance_ratio(data, threshold=0.9, plot=True):
    num_features = data.shape[1]
    pca = PCA(n_components=num_features)
    pca.fit(data)
    variance_ratio = pca.explained_variance_ratio_
    num_components = range(1, num_features + 1)

    if plot:
        plt.plot(num_components, np.cumsum(variance_ratio))
        for i in range(10, len(num_components), 10):
            plt.axvline(x=i, color='gray', linestyle='--')
            plt.text(i, np.cumsum(variance_ratio)[i-1], f'{np.cumsum(variance_ratio)[i-1]:.2f}', ha='center', va='bottom')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('Explained Variance vs. Number of Components')
        plt.show()

    return find_first_component(variance_ratio, threshold)



def main(args):
    if args.track_run:
        import wandb
    args.device = 'cuda' if torch.cuda.is_available() and (args.device == 'cuda') else 'cpu'
    if args.track_run:
        wb_run_train = wandb.init(entity=args.entity, project=args.project_name, group=args.group,
                                  # save_code=True, # Pycharm complains about duplicate code fragments
                                  job_type=args.job_type,
                                  tags=args.tags,
                                  name=f'{args.model}_train_pca_{args.pca}_var_{args.variance}' if args.use_variance_threshold else f'{args.model}_train_pca_{args.pca}_components_{args.pca_components}',
                                  config=args,
                                  )
    # graph_data = load_data(args)
    # train_data, val_data, test_data = split_data(graph_data, args.val_split, args.test_split)
    train_data, val_data, test_data = load_data(args)

    train_data: HeteroData

    if args.standardize_edge_features:
        standard_scaler_edge = StandardScaler()

        edge_attr = train_data['customer','buys','product'].edge_label[:,1:]
        train_data['customer','buys','product'].edge_label[:,1:] = torch.from_numpy(standard_scaler_edge.fit_transform(edge_attr)).float()
        edge_attr = train_data['product','rev_buys','customer'].edge_label[:,1:]
        train_data['product', 'rev_buys', 'customer'].edge_label[:,1:] = torch.from_numpy(standard_scaler_edge.transform(edge_attr)).float()

        edge_attr = val_data['customer','buys','product'].edge_label[:,1:]
        val_data['customer','buys','product'].edge_label[:,1:] = torch.from_numpy(standard_scaler_edge.transform(edge_attr)).float()
        edge_attr = val_data['product','rev_buys','customer'].edge_label[:,1:]
        val_data['product', 'rev_buys', 'customer'].edge_label[:,1:] = torch.from_numpy(standard_scaler_edge.transform(edge_attr)).float()

        edge_attr = test_data['customer','buys','product'].edge_label[:,1:]
        test_data['customer','buys','product'].edge_label[:,1:] = torch.from_numpy(standard_scaler_edge.transform(edge_attr)).float()
        edge_attr = test_data['product','rev_buys','customer'].edge_label[:,1:]
        test_data['product', 'rev_buys', 'customer'].edge_label[:,1:] = torch.from_numpy(standard_scaler_edge.transform(edge_attr)).float()

    
    # Standardize product features

    num_features = 187 
    if args.standardize_product_features:
        scaler = StandardScaler()

        # Extract non-embedding features (first 187 columns assumed to be non-embedding features)
        train_movie_features = train_data['product'].x[:, :num_features].numpy()
        scaler.fit(train_movie_features)

        # Transform these features in the training data
        train_data['product'].x[:, :num_features] = torch.from_numpy(scaler.transform(train_movie_features)).float()

        # Apply the same transformation to the validation and test data
        val_movie_features = val_data['product'].x[:, :num_features].numpy()
        val_data['product'].x[:, :num_features] = torch.from_numpy(scaler.transform(val_movie_features)).float()

        test_movie_features = test_data['product'].x[:, :num_features].numpy()
        test_data['product'].x[:, :num_features] = torch.from_numpy(scaler.transform(test_movie_features)).float()


    # # Use only the text embeddings
    if args.pca == "embeddings":
        train_data['product'].x = train_data['product'].x[:, num_features:]
        val_data['product'].x = val_data['product'].x[:, num_features:]
        test_data['product'].x = test_data['product'].x[:, num_features:]
    # Use only the original features, without the text embeddings
    if args.pca == "features":
        train_data['product'].x = train_data['product'].x[:, :num_features]
        val_data['product'].x = val_data['product'].x[:, :num_features]
        test_data['product'].x = test_data['product'].x[:, :num_features]
    print("Number of columns in train_data['product']: ", train_data['product'].x.shape[1])
    print("Number of columns in val_data['product']: ", val_data['product'].x.shape[1])
    print("Number of columns in test_data['product']: ", test_data['product'].x.shape[1])


    if args.pca in ["features","embeddings", "combined"]:
        if args.use_variance_threshold:
            n_component_with_desired_variance = plot_variance_ratio(train_data, threshold=args.variance, plot=args.plot_variance)
        else:
            n_component_with_desired_variance = args.pca_components
        print("Using n components: ", n_component_with_desired_variance)
        pca = PCA(n_components=n_component_with_desired_variance)
        pca.fit(train_data['product'].x)
        variance_ratio = pca.explained_variance_ratio_
        print("Variance ratio sum:", variance_ratio.sum())

        train_data['product'].x = torch.from_numpy(pca.transform(train_data['product'].x)).float()
        val_data['product'].x = torch.from_numpy(pca.transform(val_data['product'].x)).float()
        test_data['product'].x = torch.from_numpy(pca.transform(test_data['product'].x)).float()

    if args.pca == "separated":
        # Doing separate PCA for the text embeddings and the non-embedding features
        if args.use_variance_threshold:
            n_component_with_desired_variance = plot_variance_ratio(train_data['product'].x[:,:num_features], threshold=args.variance, plot=args.plot_variance)
        else:
            n_component_with_desired_variance = args.pca_components
        print("Using n components for features: ", n_component_with_desired_variance)
        pca_features = PCA(n_components=n_component_with_desired_variance)
        pca_features.fit(train_data['product'].x[:,:num_features])
        variance_ratio = pca_features.explained_variance_ratio_
        print("Variance ratio sum features:", variance_ratio.sum())

        if args.use_variance_threshold:
            n_component_with_desired_variance = plot_variance_ratio(train_data['product'].x[:,num_features:], threshold=args.variance, plot=args.plot_variance)
        else:
            n_component_with_desired_variance = args.pca_components
        print("Using n components for embeddings: ", n_component_with_desired_variance)
        pca_embeddings = PCA(n_components=n_component_with_desired_variance)
        pca_embeddings.fit(train_data['product'].x[:,num_features:])
        variance_ratio = pca_embeddings.explained_variance_ratio_
        print("Variance ratio sum embeddings:", variance_ratio.sum())

        # transformed_features = pca_features.transform(train_data['product'].x[:, :num_features])
        # print(transformed_features.shape)  # Check the shape after transformation
        # print(train_data['product'].x[:, :num_features].shape)  # Check the original tensor's shape


        # For training data
        features_train = torch.from_numpy(pca_features.transform(train_data['product'].x[:, :num_features])).float()
        embeddings_train = torch.from_numpy(pca_embeddings.transform(train_data['product'].x[:, num_features:])).float()
        train_data['product'].x = torch.cat((features_train, embeddings_train), dim=1)
        print(train_data['product'].x.shape)

        # For validation data
        features_val = torch.from_numpy(pca_features.transform(val_data['product'].x[:, :num_features])).float()
        embeddings_val = torch.from_numpy(pca_embeddings.transform(val_data['product'].x[:, num_features:])).float()
        val_data['product'].x = torch.cat((features_val, embeddings_val), dim=1)

        # For test data
        features_test = torch.from_numpy(pca_features.transform(test_data['product'].x[:, :num_features])).float()
        embeddings_test = torch.from_numpy(pca_embeddings.transform(test_data['product'].x[:, num_features:])).float()
        test_data['product'].x = torch.cat((features_test, embeddings_test), dim=1)

    print("Number of columns in train_data['product']: ", train_data['product'].x.shape[1])
    print("Number of columns in val_data['product']: ", val_data['product'].x.shape[1])
    print("Number of columns in test_data['product']: ", test_data['product'].x.shape[1])

    



    # print(train_data['product'].x)


    # ============
    # BATCH SETUP
    # ===========
    edge_label_index = train_data['customer', 'buys', 'product'].edge_label_index
    edge_label = train_data['customer', 'buys', 'product'].edge_label

    train_loader = LinkNeighborLoader(
        train_data.to(args.device),
        num_neighbors=[15]*3,
        batch_size=args.batch_size,
        edge_label_index=(('customer', 'buys', 'product'), edge_label_index),
        edge_label=edge_label,
        shuffle=True,
        # num_workers=4
    )

    val_loader = LinkNeighborLoader(
        val_data.to(args.device),
        num_neighbors=[10]*3,  # You might choose to have fewer layers or fewer neighbors
        batch_size=100,
        edge_label_index=(('customer', 'buys', 'product'), val_data['customer', 'buys', 'product'].edge_label_index),
        edge_label=val_data['customer', 'buys', 'product'].edge_label,
        shuffle=False  # Typically, we do not shuffle validation data
    )

    test_loader = LinkNeighborLoader(
        test_data.to(args.device),
        num_neighbors=[10]*3,  # Adjust the number of neighbors according to the model's requirements
        batch_size=100,  # Adjust the batch size based on GPU capacity and dataset size
        edge_label_index=(('customer', 'buys', 'product'), test_data['customer', 'buys', 'product'].edge_label_index),
        edge_label=test_data['customer', 'buys', 'product'].edge_label,
        shuffle=False  # Shuffling is not necessary for test data
    )

    # We have an unbalanced dataset with many labels for rating 3 and 4, and very
    # few for 0 and 1, therefore we use a weighted MSE loss.
    if args.use_weighted_loss:
        weight = torch.bincount(train_data['customer', 'product'].edge_label[:,0].long())
        weight = weight.max() / weight
        weight.to(args.device)
    else:
        weight = None
    if args.model == 'graph_sage':
        model = Model(hidden_channels=args.hidden_channels, out_channels=args.out_channels, edge_features=1, metadata=train_data.metadata())
    elif args.model == 'meta_sage':
        model = MetaSage(train_data['customer'].num_nodes, hidden_channels=args.hidden_channels, out_channels=args.out_channels)
    elif args.model == 'meta_gatv2':
        model = MetaGATv2(train_data['customer'].num_nodes, hidden_channels=args.hidden_channels, out_channels=args.out_channels, edge_channels=args.edge_channels)
    elif args.model == 'graph_transformer':
        model = MetaTransformerGat(train_data['customer'].num_nodes, hidden_channels=args.hidden_channels, out_channels=args.out_channels)
    model.to(args.device)

    # Due to lazy initialization, we need to run one model step so the number
    # of parameters can be inferred:
    # with torch.no_grad():
        # if args.model == 'graph_sage':
            # model.encoder(train_data.x_dict.to(args.device), train_data.edge_index_dict.to(args.device))

    # ========================
    # OPTIMIZER AND SETUP DATA
    # ========================
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        # Scheduler options
    # parser.add_argument('-sch', '--scheduler', type=str.lower, default='cycliclr',
    #                     choices=scheduler_choices.keys(),
    #                     help=f'Optimizer to be used {scheduler_choices.keys()}')
    # parser.add_argument('-base_lr', '--base_lr', type=float, default=3e-4,
    #                     help="Base learning rate for scheduler")
    # parser.add_argument('-max_lr', '--max_lr', type=float, default=0.001,
    #                     help="Max learning rate for scheduler")
    # parser.add_argument('-step_size_up', '--step_size_up', type=int, default=0,
    #                     help="CycleLR scheduler: step size up. If 0, then it is automatically calculated.")
    # parser.add_argument('-cyc_mom', '--cycle_momentum', type=bool, default=False,
    #                     help="CyclicLR scheduler: cycle momentum in scheduler")
    # parser.add_argument('-sch_m', '--scheduler_mode', type=str, default="triangular2",
    #                     choices=['triangular', 'triangular2', 'exp_range'],
    #                     help=f"CyclicLR scheduler: mode {['triangular', 'triangular2', 'exp_range']}")
    if args.step_size_up <= 0:
        args.step_size_up = len(train_loader.dataset) // args.batch_size
    print(args.step_size_up)
    scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimizer=optimizer,
        base_lr=args.base_lr,
        max_lr=args.max_lr,
        step_size_up=args.step_size_up,
        mode=args.scheduler_mode,
        cycle_momentum=False,
        # gamma=0.9, 
    )

    best_model_loss = np.Inf
    best_model_path = None
    for epoch in range(0, args.no_epochs):
        loss = train(model, train_loader, optimizer, weight, scheduler, args, wandb)
        # train_rmse = test(model, train_loader)
        # val_rmse = test(model, val_loader)
        val_rmse = old_test(model, val_data.to(args.device))
        if args.track_run:
            wb_run_train.log({'train_epoch_loss': loss, 
                            #   'train_epoch_rmse': train_rmse,
                              'val_epoch_rmse': val_rmse})
        print(f'Epoch: {epoch + 1:03d}, Loss: {loss:.4f}, '
            #   f'Train: {train_rmse:.4f}, '
              f'Val: {val_rmse:.4f}')
        if val_rmse < best_model_loss:
            best_model_loss = val_rmse
            Path(f'../experiments/{args.group}').mkdir(exist_ok=True, parents=True)
            new_best_path = os.path.join(f'../experiments/{args.group}',
                                         f'train-{args.group}-{args.model}-epoch{epoch + 1}'
                                         f'-loss{val_rmse:.4f}.pt')
            torch.save(model.state_dict(), new_best_path)
            if best_model_path:
                os.remove(best_model_path)
            best_model_path = new_best_path
    if args.track_run:
        wb_run_train.finish()

    args.job_type = "eval"
    if args.track_run:
        wb_run_eval = wandb.init(entity=args.entity, project=args.project_name, group=args.group,
                                 # save_code=True, # Pycharm complains about duplicate code fragments
                                 job_type=args.job_type,
                                 tags=args.tags,
                                 name=f'{args.model}_eval_pca_{args.pca}_var_{args.variance}' if args.use_variance_threshold else f'{args.model}_eval_pca_{args.pca}_components_{args.pca_components}',
                                 config=args,
                                 )
    if args.model == 'graph_sage':
        model = Model(hidden_channels=args.hidden_channels, out_channels=args.out_channels, edge_features=1, metadata=train_data.metadata())
    elif args.model == 'meta_sage':
        model = MetaSage(train_data['customer'].num_nodes, hidden_channels=args.hidden_channels, out_channels=args.out_channels)
    elif args.model == 'meta_gatv2':
        model = MetaGATv2(train_data['customer'].num_nodes, hidden_channels=args.hidden_channels, out_channels=args.out_channels, edge_channels=args.edge_channels)
    elif args.model == 'graph_transformer':
        model = MetaTransformerGat(train_data['customer'].num_nodes, hidden_channels=args.hidden_channels, out_channels=args.out_channels)
    model.load_state_dict(torch.load(best_model_path))
    model.to(args.device)
    # test_rmse = test(model, test_loader)
    test_rmse = old_test(model, test_data.to(args.device))
    if args.track_run:
        wb_run_eval.log({'test_rmse': test_rmse})
        wb_run_eval.finish()


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument('--use_weighted_loss', action='store_true', default=False,
                        help='Whether to use weighted MSE loss.')
    PARSER.add_argument('--no_epochs', default=20, type=int)
    # Wandb logging options
    PARSER.add_argument('-entity', '--entity', type=str, default="njmarko",
                        help="Name of the team. Multiple projects can exist for the same team.")
    PARSER.add_argument('-project_name', '--project_name', type=str, default="graph-recommendation-movielens",
                        help="Name of the project. Each experiment in the project will be logged separately"
                             " as a group")
    PARSER.add_argument('-group', '--group', type=str, default="paper",
                        help="Name of the experiment group. Each model in the experiment group will be logged "
                             "separately under a different type.")
    PARSER.add_argument('-save_model_wandb', '--save_model_wandb', type=bool, default=True,
                        help="Save best model to wandb run.")
    PARSER.add_argument('-job_type', '--job_type', type=str, default="train",
                        help="Job type {train, eval}.")
    PARSER.add_argument('-tags', '--tags', nargs="*", type=str, default="train",
                        help="Add a list of tags that describe the run.")
    # Model options
    model_choices = ['graph_sage', 'meta_sage', 'meta_gatv2', 'graph_transformer']

    PARSER.add_argument('-m', '--model', type=str.lower, default="meta_sage",
                        choices=model_choices,
                        help=f"Model to be used for training {model_choices}")
    PARSER.add_argument('--hidden_channels', default=64, type=int)
    PARSER.add_argument('--out_channels', default=64, type=int)
    PARSER.add_argument('--edge_channels', default=5, type=int)
    # Training options
    PARSER.add_argument('-device', '--device', type=str, default='cuda', help="Device to be used")
    PARSER.add_argument('--val_split', default=0.15, type=float)
    PARSER.add_argument('--test_split', default=0.15, type=float)
    PARSER.add_argument('--word_embeddings', action='store_true', default=True, help='Use movie synopsis word embeddings')

    # Optimizer and scheduler options
    PARSER.add_argument('--lr', default=3e-4)
    PARSER.add_argument('--weight_decay', default=0.05)
    PARSER.add_argument('--base_lr', default=5e-3, type=float)
    PARSER.add_argument('--max_lr', default=5e-2, type=float)
    PARSER.add_argument('-sch_m', '--scheduler_mode', type=str, default="triangular2",
                    choices=['triangular', 'triangular2', 'exp_range'],
                    help=f"CyclicLR scheduler: mode {['triangular', 'triangular2', 'exp_range']}")
    PARSER.add_argument('-step_size_up', '--step_size_up', type=int, default=0,
                        help="CycleLR scheduler: step size up. If 0, then it is automatically calculated.")

    PARSER.add_argument('--track_run', action='store_true', default=True, help='Track run on wandb')

    # Batch options
    PARSER.add_argument('--batch_size', default=128)
    PARSER.add_argument('--num_partitions', default=150)

    # Standardization
    PARSER.add_argument('--standardize_product_features', action='store_true', default=True, help='Standardize product features')
    PARSER.add_argument('--standardize_edge_features', action='store_true', default=True, help='Standardize edge features')

    # PCA options
    PARSER.add_argument('--pca', type=str.lower, default="separated",
                        choices=["none", "features", "embeddings", "combined", "separated"],
                        help="PCA options for feature reduction")
    PARSER.add_argument('--pca_components', type=int, default=32,
                        help="Number of PCA components to keep")
    PARSER.add_argument('--use_variance_threshold', action='store_true', default=True, help='Use variance threshold for PCA')
    PARSER.add_argument('--variance', type=float, default=0.85, help='Variance threshold for PCA')
    PARSER.add_argument('--plot_variance', action='store_true', default=False, help='Plot variance ratio')
    

    ARGS = PARSER.parse_args()
    main(ARGS)

    
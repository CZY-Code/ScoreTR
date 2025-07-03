import os
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import scipy
from sklearn.model_selection import KFold

def get_ob_data(ob_tensor, ori_mask, batch_size=4096):
    H, W, C = ob_tensor.shape
    # norm_factors = torch.tensor([H, W, C], device=ob_tensor.device, dtype=torch.float32)

    train_ind = torch.argwhere(ori_mask == 0)  # [N, 3] 格式为(h,w,c)
    # train_ind = train_ind.float() / norm_factors
    train_val = ob_tensor[ori_mask == 0]       # [N,]

    test_ind = torch.argwhere(ori_mask == 1)  # [N, 3] 格式为(h,w,c)
    # test_ind = test_ind.float() / norm_factors
    test_val = ob_tensor[ori_mask == 1]       # [N,]

    train_dt = TensorDataset(train_ind, train_val)
    test_dt = TensorDataset(test_ind, test_val)

    train_loader = DataLoader(train_dt, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dt, batch_size=2048, shuffle=False)

    return {'train': train_loader, 'test': test_loader}


def tensor2data(ob_tensor, batch_size=4096):
    H, W, C = ob_tensor.shape
    h_indices = torch.arange(H)  # 深度方向索引
    w_indices = torch.arange(W)  # 高度方向索引
    c_indices = torch.arange(C)  # 宽度方向索引
    # 使用 torch.meshgrid 生成网格索引
    h_grid, w_grid, c_grid = torch.meshgrid(h_indices, w_indices, c_indices, indexing='ij')
    # 将网格索引展平并堆叠为 [HWC, 3] 的张量
    indices = torch.stack([h_grid.flatten(), w_grid.flatten(), c_grid.flatten()], dim=1)
    # 将原始张量展平为 [HWC, ] 的张量
    values = ob_tensor.flatten()
    data = TensorDataset(indices, values)
    train_loader = DataLoader(data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(data, batch_size=batch_size, shuffle=False)

    return {'train': train_loader, 'test': test_loader}


def get_air_data(file_path: str, batch_size=int(2**10)):
    data_loaders = []
    dt = np.load(os.path.join(file_path, 'beijing.npy'), allow_pickle=True)
    dt = dt.item()
    train_dt_raw = dt['train_folds']
    test_dt_raw = dt['test_folds']

    for i in range(5):
        train_ind = train_dt_raw[i][:, :2].astype(int)
        train_time = train_dt_raw[i][:, 2:3].astype(float).reshape(-1) / 10.
        train_val = train_dt_raw[i][:, -1].astype(float).reshape(-1) / 5.

        test_ind = test_dt_raw[i][:, :2].astype(int)
        test_time = test_dt_raw[i][:, 2:3].astype(float).reshape(-1) / 10.
        test_val = test_dt_raw[i][:, -1].astype(float).reshape(-1) / 5.

        train_ind = torch.tensor(train_ind, dtype=torch.int32)
        test_ind = torch.tensor(test_ind, dtype=torch.int32)

        train_val = torch.tensor(train_val, dtype=torch.float32)
        test_val = torch.tensor(test_val, dtype=torch.float32)

        train_time = torch.tensor(train_time, dtype=torch.float32)
        test_time = torch.tensor(test_time, dtype=torch.float32)

        train_dt = TensorDataset(train_ind, train_time, train_val)
        test_dt = TensorDataset(test_ind, test_time, test_val)

        train_loader = DataLoader(train_dt, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dt, batch_size=2000, shuffle=False)

        data_loaders.append({'train': train_loader, 'test': test_loader})

    return data_loaders


def get_click_data(file_path: str, batch_size=int(2**10)):
    data_loaders = []
    dt = np.load(os.path.join(file_path, 'ctr_50k.npy'), allow_pickle=True).item()
    n_node = dt['ndims'] #[7, 2842, 4127]
    data = dt['data']

    for i in range(5):
        train_ind = data[i]['tr_ind'].astype(int)
        train_time = data[i]['tr_T'].astype(float).reshape(-1)
        train_val = data[i]['tr_y'].astype(float).reshape(-1) / 10.0 #(40000,)

        test_ind = data[i]['te_ind'].astype(int)
        test_time = data[i]['te_T'].astype(float).reshape(-1)
        test_val = data[i]['te_y'].astype(float).reshape(-1) / 10.0 #(10000,) 
        
        train_ind = torch.tensor(train_ind, dtype=torch.int32)
        test_ind = torch.tensor(test_ind, dtype=torch.int32)

        train_val = torch.tensor(train_val, dtype=torch.float32)
        test_val = torch.tensor(test_val, dtype=torch.float32)

        train_time = torch.tensor(train_time, dtype=torch.float32)
        test_time = torch.tensor(test_time, dtype=torch.float32)

        train_dt = TensorDataset(train_ind, train_time, train_val)
        test_dt = TensorDataset(test_ind, test_time, test_val)

        train_loader = DataLoader(train_dt, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dt, batch_size=1000, shuffle=False)

        data_loaders.append({'train': train_loader, 'test': test_loader})

    return data_loaders

def get_alog_data(data_path, batch_size=int(2**10)):
    data_loaders = []
    val_max = 17.0190
    val_min = 0.6931
    scale = val_max

    def load_txt(file_path):
        ind = []
        val = []
        with open(file_path, 'r') as f:
            for line in f:
                items = line.strip().split(',')
                val.append(float(items[-1]))
                ind.append([int(idx)-1 for idx in items[0:-1]])
            ind = np.array(ind).astype(int)
            val = np.array(val).astype(float) / scale
        return ind, val

    for i in range(5):
        train_ind, train_val = load_txt(os.path.join(data_path, f'alog/train-fold-{i+1}.txt'))
        test_ind, test_val = load_txt(os.path.join(data_path, f'alog/test-fold-{i+1}.txt'))

        # train: (10538, 3) (10538,) test: (2634, 3) (2634,)
        train_ind = torch.tensor(train_ind, dtype=torch.int32)
        test_ind = torch.tensor(test_ind, dtype=torch.int32)

        train_val = torch.tensor(train_val, dtype=torch.float32)
        test_val = torch.tensor(test_val, dtype=torch.float32)
        
        train_dt = TensorDataset(train_ind, train_val)
        test_dt = TensorDataset(test_ind, test_val)
        
        train_loader = DataLoader(train_dt, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dt, batch_size=2634, shuffle=False)
        
        data_loaders.append({'train': train_loader, 'test': test_loader})

    return data_loaders

def get_acc_data(data_path, batch_size=int(2**10)):
    data_loaders = []
    test_loaders = []
    val_max = 17.4271
    val_min = 0.6931
    scale = val_max

    train_data = np.loadtxt(os.path.join(data_path, 'acc/ibm-large-tensor.txt')) #(1128092, 4)
    # nonzero_ind = np.nonzero(train_data[:,-1])[0] #(564046,)
    # train_data = train_data[nonzero_ind, :]
    train_ind, train_val = train_data[:, :-1].astype(int), train_data[:, -1].astype(float) / scale
    train_ind = torch.tensor(train_ind, dtype=torch.int32)
    train_val = torch.tensor(train_val, dtype=torch.float32)
    train_dt = TensorDataset(train_ind, train_val)
    train_loader = DataLoader(train_dt, batch_size=batch_size, shuffle=True)

    test_data = scipy.io.loadmat(os.path.join(data_path, 'acc/ibm.mat'))['data']
    # train_ind = data['Y'][0,0]['subs'][0,0].astype(int) - 1 #(564046, 3)
    # train_val = data['Y'][0,0]['vals'][0,0][:,0].astype(float) / scale #(564046, 1)
    # print(data.dtype.names) #('nzind', 'test_ind_all', 'size', 'test', 'Y')
    # print(test_data['test_ind_all'][0,0].shape) (141012, 3)
    for i in range(50):
        test_ind = test_data['test'][0,0][0,i]['subs'][0,0].astype(int) - 1 #(2000, 3)
        test_val = test_data['test'][0,0][0,i]['Ymiss'][0,0][:,0].astype(float) / scale #(2000,)
        test_ind = torch.tensor(test_ind, dtype=torch.int32)
        test_val = torch.tensor(test_val, dtype=torch.float32)
        test_dt = TensorDataset(test_ind, test_val)
        test_loader = DataLoader(test_dt, batch_size=2000, shuffle=False)
        test_loaders.append(test_loader)

    data_loaders.append({'train': train_loader, 'test': test_loaders})

    return data_loaders


def get_sim_data(dist='beta', batch_size=int(2**10), R=5, shape=(8, 8)):
    I1, I2 = shape
    N = 200
    Z = torch.distributions.Uniform(0.0, 1.0)
    z1 = Z.sample((I1, R)).cuda()
    z2 = Z.sample((R, I2)).cuda()
    if dist == 'beta':
        alpha = z1 @ z2
        beta_dist = torch.distributions.Beta(alpha, 5.0)
        data_val = beta_dist.sample((N,)).cuda()  # [10000, 1]
    elif dist == 'MoG':
        mu = z1 @ z2
        mu_1 = torch.cos(mu)
        mu_2 = torch.sin(mu)
        choices = torch.bernoulli(torch.full((N, I1, I2), 0.6)).long().cuda()  # [200, 8, 8]
        normal_dist_1 = torch.distributions.Normal(mu_1, 0.1)
        normal_dist_2 = torch.distributions.Normal(mu_2, 0.25)
        samples_1 = normal_dist_1.sample((N,)).cuda()
        samples_2 = normal_dist_2.sample((N,)).cuda()
        # data_val = 0.6 * normal_dist_1.sample((N,)).cuda() + 0.4 * normal_dist_2.sample((N,)).cuda()
        data_val = torch.where(choices == 0, samples_1, samples_2)
    elif dist == 'exp':
        lambda_param = z1 @ z2
        exp_dist = torch.distributions.Exponential(lambda_param)
        data_val = exp_dist.sample((N,)).cuda()
    else:
        raise ValueError(f"Unsupported distribution: {dist}")
    h_idx = torch.arange(I1, device='cuda').view(1, 1, I1, 1)  # [1, 1, 8, 1]
    w_idx = torch.arange(I2, device='cuda').view(1, 1, 1, I2)  # [1, 1, 1, 8]
    h_idx = h_idx.expand(N, 1, I1, I2)  # [200, 1, 8, 8]
    w_idx = w_idx.expand(N, 1, I1, I2)  # [200, 1, 8, 8]
    data_ind = torch.cat([h_idx, w_idx], dim=1)  # [200, 2, 8, 8]
    data_ind = data_ind.permute(0, 2, 3, 1).reshape(-1, 2)  # [12800, 2]
    data_val = data_val.view(-1)  # [12800,]

    train_dt = TensorDataset(data_ind, data_val)
    train_loader = DataLoader(train_dt, batch_size=batch_size, shuffle=True)

    # test_sample(train_dt, idx=[4, 4], dist=dist)  # 可视化样本分布
    return train_loader, train_dt, (z1, z2)

def test_sample(dataset, idx=[4, 4], dist='beta'):
    from scipy.stats import beta, gaussian_kde, norm
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    if dist == 'beta':
        val_max, val_min = 1.0, 0.0
    elif dist == 'MoG':
        val_max, val_min = 1.1, -0.4
    elif dist == 'exp':   
        val_max, val_min = 1.0, 0.0
    
    x_val_tot = torch.stack([dataset[i*64 + idx[0]*8 + idx[1]][1] for i in range(200)]) #[200]
    x_val_tot = x_val_tot.cpu().detach().numpy()
    val_kde = gaussian_kde(x_val_tot)
    x_range = np.linspace(val_min, val_max, 500)
    val_density = val_kde(x_range)

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.set_facecolor('#f0f0f0')  # 浅灰背景
    ax.plot(x_range, val_density, color='#f27830', linewidth=2, label=r'Obs')
    ax.fill_between(x_range, val_density, alpha=0.2, color='#fcd5b4')
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5)) # x轴主网格间隔0.5
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax.set_xlabel('x', fontsize=10)
    ax.set_ylabel('Density', fontsize=10)

    # 其他设置
    ax.legend(fontsize=9, loc='upper right')  # 建议指定图例位置
    ax.grid(True, linestyle='--', alpha=0.8, color='#ffffff')  # 降低网格透明度，避免喧宾夺主
    fig.tight_layout() 
    plt.show()


def get_continuous_data(batch_size=int(2**10), shape=(8, 8)):
    def omega_function(t):
        omega11 = torch.sin(2 * torch.pi * t)
        omega12 = torch.cos(2 * torch.pi * t)
        omega21 = torch.pow(torch.sin(2 * torch.pi * t), 2)
        omega22 = torch.pow(torch.sin(5 * torch.pi * t), 2) * torch.cos(5 * torch.pi * t)
        omega_matrix = torch.stack([omega11, omega12, omega21, omega22], dim=1).reshape(-1, 2, 2)
        return  omega_matrix
    
    def create_mask(length, intervals):
        mask = torch.zeros(length, dtype=torch.bool)
        for start_ratio, end_ratio in intervals:
            start_idx = int(length * start_ratio)
            end_idx = int(length * end_ratio)
            mask[start_idx:end_idx] = True
        return mask

    def generate_random_intervals(num_intervals=3, interval_width=0.1, min_gap=0.05, max_attempts=100):
        rng = np.random.default_rng(2)
        max_start = 1.0 - interval_width
        for attempt in range(max_attempts):
            # 生成随机起始点并排序
            starts = rng.uniform(0, max_start, size=num_intervals)
            starts.sort()
            
            # 检查所有相邻间隔是否满足条件
            valid = True
            for i in range(1, num_intervals):
                if starts[i] - starts[i-1] < min_gap + interval_width:
                    valid = False
                    break
                    
            if valid:
                # 生成测试区间
                test_list = [(start, start + interval_width) for start in starts]
                # 计算训练区间
                train_list = []
                prev_end = 0.0
                for start, end in test_list:
                    if prev_end < start:
                        train_list.append((prev_end, start))
                    prev_end = end
                
                if prev_end < 1.0:
                    train_list.append((prev_end, 1.0))
                return train_list, test_list
        raise ValueError(f"尝试 {max_attempts} 次后仍无法生成满足条件的区间")

    I1, I2 = shape
    N = 200
    scale = 10.0
    mu_1 = torch.tensor([0.0, 2.0])  # 均值
    mu_2 = torch.tensor([1.0, 1.0])  # 均值
    sigma = 2 * torch.eye(2)
    Z1 = torch.distributions.MultivariateNormal(mu_1, sigma)
    Z2 = torch.distributions.MultivariateNormal(mu_2, sigma)
    z1 = Z1.sample((I1,)) # [8, 2]
    z2 = Z2.sample((I2,)) # [8, 2]
    t = torch.linspace(0, 1, N)  # [200,]
    omega = omega_function(t)  # [200, 2, 2]

    # train_list = [(0, 0.15), (0.25, 0.50), (0.70, 0.85), (0.95, 1.0)]
    train_list, test_list = generate_random_intervals(num_intervals=12, interval_width=0.05, min_gap=0.01, max_attempts=10000000)
    mask = create_mask(N, train_list)

    data_val = torch.einsum('ia, jb, tab -> tij', z1, z2, omega)
    test_val = data_val.clone().reshape(-1) / scale
    train_val = data_val[mask]
    train_val = train_val.reshape(-1) / scale # [12800,]

    h_idx = torch.arange(I1, device='cuda').view(1, 1, I1, 1)  # [1, 1, 8, 1]
    w_idx = torch.arange(I2, device='cuda').view(1, 1, 1, I2)  # [1, 1, 1, 8]
    h_idx = h_idx.expand(N, 1, I1, I2)  # [200, 1, 8, 8]
    w_idx = w_idx.expand(N, 1, I1, I2)  # [200, 1, 8, 8]

    data_ind = torch.cat([h_idx, w_idx], dim=1)  # [200, 2, 8, 8]
    test_ind = data_ind.clone().permute(0, 2, 3, 1).reshape(-1, 2)
    train_ind = data_ind[mask]
    train_ind = train_ind.permute(0, 2, 3, 1).reshape(-1, 2)  # [12800, 2]

    data_time = t.reshape(-1, 1, 1).expand(-1, I1, I2)  # [200, 8, 8]
    test_time = data_time.clone().reshape(-1)
    train_time = data_time[mask]
    train_time = train_time.reshape(-1)

    train_dt = TensorDataset(train_ind, train_time, train_val)
    test_dt = TensorDataset(test_ind, test_time, test_val)
    train_loader = DataLoader(train_dt, batch_size=batch_size, shuffle=True)
    
    return train_loader, test_dt, test_list
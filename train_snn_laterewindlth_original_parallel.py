import time
import time
import utils
import config_lth
import torch.cuda.amp as amp
import torchvision
import os
import copy
import pickle
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as m
from torch.nn.parallel import DistributedDataParallel as DDP


from archs.cifarsvhn.vgg import vgg16_bn
from archs.cifarsvhn.resnet import ResNet19
from archs.fmnist.vgg import vgg16_bn as fmnist_vgg16_bn
from archs.fmnist.resnet import ResNet19 as fmnist_ResNet19

from utils_for_snn_lth import *
from utils import data_transforms
from spikingjelly.clock_driven.functional import reset_net

def main():
    cudnn.benchmark = True
    cudnn.deterministic = True
    args = config_lth_parallel.get_args()
    # define dataset
    train_transform, valid_transform = data_transforms(args)
    if args.dataset == 'fmnist':
        trainset = torchvision.datasets.FashionMNIST(root=os.path.join(args.data_dir, 'fmnist'), train=True,
                                                download=True, transform=train_transform)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                                   shuffle=True, pin_memory=True, num_workers=4)
        valset = torchvision.datasets.FashionMNIST(root=os.path.join(args.data_dir, 'fmnist'), train=False,
                                              download=True, transform=valid_transform)
        val_loader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size,
                                                 shuffle=False, pin_memory=True, num_workers=4)
        n_class = 10
    elif args.dataset == 'svhn':
        trainset = torchvision.datasets.SVHN(root=os.path.join(args.data_dir, 'svhn'), split='train',
                                                     download=True, transform=train_transform)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                                   shuffle=True, pin_memory=True, num_workers=4)
        valset = torchvision.datasets.SVHN(root=os.path.join(args.data_dir, 'svhn'), split='test',
                                                   download=True, transform=valid_transform)
        val_loader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size,
                                                 shuffle=False, pin_memory=True, num_workers=4)
        n_class = 10

    elif args.dataset == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(root=os.path.join(args.data_dir, 'cifar10'), train=True,
                                                download=True, transform=train_transform)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                                   shuffle=True, pin_memory=True, num_workers=4)
        valset = torchvision.datasets.CIFAR10(root=os.path.join(args.data_dir, 'cifar10'), train=False,
                                              download=True, transform=valid_transform)
        val_loader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size,
                                                 shuffle=False, pin_memory=True, num_workers=4)
        n_class = 10
    elif args.dataset == 'cifar100':
        trainset = torchvision.datasets.CIFAR100(root=os.path.join(args.data_dir, 'cifar100'), train=True,
                                                download=True, transform=train_transform)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                                   shuffle=True, pin_memory=True, num_workers=4)
        valset = torchvision.datasets.CIFAR100(root=os.path.join(args.data_dir, 'cifar100'), train=False,
                                              download=True, transform=valid_transform)
        val_loader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size,
                                                 shuffle=False, pin_memory=True, num_workers=4)
        n_class =100



    criterion = nn.CrossEntropyLoss()

    if  args.dataset != 'fmnist' and args.arch == 'vgg16':
        model = vgg16_bn(num_classes=n_class, total_timestep=args.timestep).cuda()
    elif args.dataset != 'fmnist' and args.arch == 'resnet19':
        model = ResNet19(num_classes=n_class, total_timestep=args.timestep).cuda()
    elif args.dataset == 'fmnist' and args.arch == 'vgg16':
        model = fmnist_vgg16_bn(num_classes=n_class, total_timestep=args.timestep).cuda()
    elif args.dataset == 'fmnist' and args.arch == 'resnet19':
        model = fmnist_ResNet19(num_classes=n_class, total_timestep=args.timestep).cuda()
    else:
        exit()


    # Copying and Saving Initial State
    initial_state_dict = copy.deepcopy(model.state_dict())
    utils.checkdir(f"{os.getcwd()}/snn_laterewind_lth/{args.arch}/{args.dataset}/round{args.round}")
    torch.save(model.state_dict(), f"{os.getcwd()}/snn_laterewind_lth/{args.arch}/{args.dataset}/round{args.round}/initial_state_dict.pth.tar")

    # Making Initial Mask
    mask = make_mask(model)
    
    model = nn.DataParallel(model,device_ids=[0,1,2,3])
    # model = model.cuda()


    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, args.momentum, args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
    else:
        print ("will be added...")
        exit()

    if args.scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(args.end_iter*0.5),int(args.end_iter*0.75)], gamma=0.1)
    elif args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max= int(args.end_iter), eta_min= 0)
    else:
        print ("will be added...")
        exit()


    # Pruning
    # NOTE First Pruning Iteration is of No Compression
    best_accuracy = 0
    ITERATION = args.prune_iterations
    # ITERATION = 2
    comp = np.zeros(ITERATION, float)
    bestacc = np.zeros(ITERATION, float)
    all_loss = np.zeros(args.end_iter, float)
    all_accuracy = np.zeros(args.end_iter, float)

    rewinding_epoch = 0

    # if args.arch == 'vgg16':
    #     model.load_state_dict(torch.load('save_model/vgg/0_model.pth.tar').state_dict())
    # else:
    #     print('we do resent')
    #     model.load_state_dict(torch.load('save_model/res/0_model.pth.tar').state_dict())
    #
    # # accuracy = test(model, val_loader, criterion)
    # # print('full training accuracy:', accuracy)
    n_pe = 16
    arc_style = 'sata'
    rec_mode = 'strong'

    print(f"Number of PE: {n_pe} Archtiecture: {arc_style} Mode: {rec_mode}")


    for round_ in range(ITERATION):
        best_accuracy = 0

        if not round_ == 0:
            # Dumping mask _before
            utils.checkdir(f"{os.getcwd()}/dumps/snn_laterewind_lth_parallel/{args.arch}/{args.dataset}/{arc_style}/{n_pe}")
            with open(
                    f"{os.getcwd()}/dumps/snn_laterewind_lth_parallel/{args.arch}/{args.dataset}/{arc_style}/{n_pe}/mask_{comp1}.pkl",
                    'wb') as fp:
                pickle.dump(mask, fp)

            model, mask = prune_by_percentile(args, args.prune_percent, mask , model)


            layerwise_u_list_avg, _, _, dyn,was,lat  = utilization_operation_network(mask,model,n_PE=n_pe,operation="check",arch_style=arc_style)

            print("Original LTH without recovering...")

            print("avg overall Layerwise Utilization: ")
            print(sum(layerwise_u_list_avg) / len(layerwise_u_list_avg))

            print("avg Layerwise Utilization: ")
            print(layerwise_u_list_avg)

            print(f"Total Dynamic: {dyn} Total Wasted: {was} Total Latency: {lat}")


            # new_mask = utilization_operation_network(copy.deepcopy(mask), model, n_PE=n_pe, operation="recover",arch_style=arc_style,mode=rec_mode)

            # layerwise_u_list_avg, _, _, dyn,was,lat = utilization_operation_network(new_mask,model,n_PE=n_pe,operation="check",arch_style=arc_style)

            # print("Origina LTH but with oneshot...")

            # print("avg overall Layerwise Utilization: ")
            # print(sum(layerwise_u_list_avg) / len(layerwise_u_list_avg))

            # print("avg Layerwise Utilization: ")
            # print(layerwise_u_list_avg)

            # print("Total Dynamic: ")
            # print(dyn)

            # print("Total Wasted: ")
            # print(was)

            # print("Total Latency: ")
            # print(lat)

            # scam_model = copy.deepcopy(model)
            # recover_only_model = original_initialization(new_mask, initial_state_dict, scam_model)

            # print("Check the prune ratio for recovered model without in loop.")

            # comp_test = utils.print_nonzeros(recover_only_model)
            # accuracy = test(recover_only_model, val_loader, criterion)
            # print("Accuracy for recovered model without in loop.")
            # print(accuracy)


            model = original_initialization(mask, initial_state_dict, model)

            optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, args.momentum, args.weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(args.end_iter- rewinding_epoch),
                                                                   eta_min= 0)


        print(f"\n--- Pruning Level [round{args.round}:{round_}/{ITERATION}]: ---")

        # Print the table of Nonzeros in each layer
        comp1 = utils.print_nonzeros(model)
        comp[round_] = comp1
        loss = 0
        accuracy =0
        for iter_ in range(args.end_iter - rewinding_epoch):
            s_time = time.time()
            # Frequency for Testing
            if (iter_+1) % args.valid_freq == 0:
                accuracy = test(model, val_loader, criterion)

                # Save Weights
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    utils.checkdir(f"{os.getcwd()}/snn_laterewind_lth_parallel/{args.arch}/{args.dataset}/{arc_style}/{n_pe}")
                    torch.save(model,
                               f"{os.getcwd()}/snn_laterewind_lth_parallel/{args.arch}/{args.dataset}/{arc_style}/{n_pe}/{round_}_model.pth.tar")

            # Training
            t1 = time.time()
            loss = train(args, iter_, train_loader, model, criterion, optimizer, scheduler)
            t2 = time.time()
            print(f'train_time{t2-t1}')
            all_loss[iter_] = loss
            all_accuracy[iter_] = accuracy


            #TODO Late rewinding init weight at 20epoch
            if round_ == 0 and iter_ == args.rewinding_epoch:
                print ('find laterewinding weight--------')
                initial_state_dict = copy.deepcopy(model.state_dict())
                rewinding_epoch = args.rewinding_epoch

            # Frequency for Printing Accuracy and Loss
            if (iter_ +1)% args.print_freq == 0:
                print(f'Train Epoch: {iter_}/{args.end_iter} Loss: {loss:.6f} Accuracy: {accuracy:.2f}% Best Accuracy: {best_accuracy:.2f}%')


        bestacc[round_]=best_accuracy
        _,spa = test_spa(model, val_loader, criterion)
        print("Round sparsity: ", spa)
        print("Round best accuracy: ", best_accuracy)


        # # Dumping mask
        # utils.checkdir(f"{os.getcwd()}/dumps/snn_laterewind_lth/{args.arch}/{args.dataset}/round{args.round}")
        # with open(f"{os.getcwd()}/dumps/snn_laterewind_lth/{args.arch}/{args.dataset}/round{args.round}/mask_{comp1}.pkl",
        #           'wb') as fp:
        #     pickle.dump(mask, fp)
    

def train(args, epoch, train_data,  model, criterion, optimizer, scheduler):
    model.train()
    EPS = 1e-6
    for batch_idx, (imgs, targets) in enumerate(train_data):
        train_loss = 0.0

        optimizer.zero_grad()
        imgs, targets = imgs.cuda(), targets.cuda()
        with amp.autocast():

            output_list = model(imgs)
            # print(len(output_list[0]))
            
            for output in output_list:
                train_loss += criterion(output, targets) / args.timestep
            
        train_loss.sum().backward()

        # Freezing Pruned weights by making their gradients Zero
        for name, p in model.named_parameters():
            if 'weight' in name:
                tensor = p.data
                if (len(tensor.size())) == 1:
                    continue
                grad_tensor = p.grad
                grad_tensor = torch.where(tensor.abs() < EPS, torch.zeros_like(grad_tensor), grad_tensor)
                p.grad.data = grad_tensor
        optimizer.step()
        reset_net(model)
    scheduler.step()

    return torch.mean(train_loss).item()

    # return train_loss.item()






if __name__ == '__main__':
    main()

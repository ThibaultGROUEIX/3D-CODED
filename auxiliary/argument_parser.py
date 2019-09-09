import argparse
import auxiliary.my_utils as my_utils
import os
import datetime


def Args2String(opt):
    my_str = ""
    for i in opt.__dict__.keys():
        if i == "model":
            if opt.__dict__[i] is None:
                my_str = my_str + str(0) + "_"
            else:
                my_str = my_str + str(1) + "_"
        else:
            my_str = my_str + str(opt.__dict__[i]) + "_"
    my_str = my_str.replace('/', '-')
    return my_str


def parser():
    parser = argparse.ArgumentParser()
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=8)
    parser.add_argument('--nepoch', type=int, default=100, help='number of epochs to train for')
    parser.add_argument('--start_epoch', type=int, default=0, help='number of epochs to train for')
    parser.add_argument('--randomize', type=int, default=0, help='if 1, projects predicted correspondences point on target mesh')
    parser.add_argument('--lrate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--lr_decay_1', type=int, default=80, help='learning rate decay 1')
    parser.add_argument('--lr_decay_2', type=int, default=90, help='learning rate decay 2')

    # Data
    parser.add_argument('--number_points', type=int, default=2500, help='Number of point sampled on the object')

    # Save dirs and reload
    parser.add_argument('--id', type=str, default="0",help='training name')
    parser.add_argument('--env', type=str, default="CODED", help='visdom environment')
    parser.add_argument('--display', type=int, default=1, help='visdom environment')
    parser.add_argument('--port', type=int, default=8889, help='visdom port')
    parser.add_argument('--dir_name', type=str, default="",  help='dirname')

    # Network
    parser.add_argument('--model', type=str, default='', help='optional reload model path')

    # Loss
    parser.add_argument(
        "--accelerated_chamfer",
        type=int,
        default=1,
        help="use custom build accelarated chamfer",
    )
    # Eval parameters
    parser.add_argument('--HR', type=int, default=1, help='Use high Resolution template')
    parser.add_argument('--LR_input', type=int, default=1, help='Use Low Resolution Input ')
    parser.add_argument('--reg_num_steps', type=int, default=3000, help='number of regression steps')
    parser.add_argument('--inputA', type=str, default="data/example_0.ply", help='your path to mesh 0')
    parser.add_argument('--inputB', type=str, default="data/example_1.ply", help='your path to mesh 1')
    parser.add_argument('--num_angles', type=int, default=100,
                        help='number of angle in the search of optimal reconstruction. Set to 1, if you mesh are already facing the cannonical direction as in data/example_1.ply')
    parser.add_argument('--clean', type=int, default=1, help='if 1, remove points that dont belong to any edges')
    parser.add_argument('--scale', type=int, default=1, help='if 1, scale input mesh to have same volume as the template')
    parser.add_argument('--project_on_target', type=int, default=0,  help='if 1, projects predicted correspondences point on target mesh')

    opt = parser.parse_args()


    opt.HR = my_utils.int_2_boolean(opt.HR)
    opt.LR_input = my_utils.int_2_boolean(opt.LR_input)
    opt.clean = my_utils.int_2_boolean(opt.clean)
    opt.scale = my_utils.int_2_boolean(opt.scale)
    opt.project_on_target = my_utils.int_2_boolean(opt.project_on_target)
    opt.randomize = my_utils.int_2_boolean(opt.randomize)
    opt.accelerated_chamfer = my_utils.int_2_boolean(opt.accelerated_chamfer)
    opt.display = my_utils.int_2_boolean(opt.display)

    opt.date = str(datetime.datetime.now())
    now = datetime.datetime.now()
    if opt.dir_name=="":
        opt.dir_name = os.path.join('log', opt.id + now.isoformat())


    # my_utils.print_arg(opt)

    return opt

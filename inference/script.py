from __future__ import print_function
import sys

sys.path.append('./auxiliary/')
sys.path.append('./training/')
sys.path.append('/app/python/')
sys.path.append('./')
import my_utils
import argparse
import os
import datetime
import correspondences


def main(opt, network):
    if not os.path.exists(os.path.join(opt.dataset_path, "test_scan_006.ply")):
        print("please download test data from http://faust.is.tue.mpg.de/")
        os.exit()

    if not os.path.exists("learning_elementary_structure_trained_models/0point_translation/network.pth"):
        os.system("chmod +x ./inference/download_trained_models.sh")
        os.system("./inference/download_trained_models.sh")

    if opt.dir_name == "":
        now = datetime.datetime.now()
        save_path = now.isoformat()
        save_path = opt.id + save_path
        dir_name = os.path.join('log_inference', save_path)
        if not os.path.exists("log_inference"):
            print("Creating log_inference folder")
            os.mkdir("log_inference")
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
    else:
        opt.model_path = f"{opt.dir_name}/network.pth"

    network.save_template_png(opt.dir_name)
    network.make_high_res_template_from_low_res()
    inf = correspondences.Inference(model_path=opt.model_path, save_path=opt.dir_name, LR_input=opt.LR_input,
                                    network=network, HR=opt.HR, reg_num_steps=opt.reg_num_steps, uniformize = opt.uniformize)

    if opt.faust == "INTER":
        FILES = "006_021.txt 011_107.txt 012_043.txt 015_098.txt 028_079.txt 033_080.txt 038_196.txt 039_142.txt 045_114.txt 046_128.txt 055_191.txt 056_077.txt 061_104.txt 065_165.txt 067_019.txt 078_171.txt 084_183.txt 089_134.txt 090_068.txt 093_010.txt 094_133.txt 095_117.txt 101_152.txt 115_199.txt 124_173.txt 129_053.txt 131_174.txt 136_002.txt 137_072.txt 146_025.txt 149_057.txt 151_184.txt 155_112.txt 156_014.txt 158_187.txt 161_103.txt 166_037.txt 167_044.txt 178_180.txt 198_029.txt"
    elif opt.faust == "INTRA":
        FILES = "000_003.txt 004_008.txt 005_001.txt 007_013.txt 009_016.txt 018_017.txt 022_020.txt 024_023.txt 026_030.txt 032_035.txt 034_031.txt 036_027.txt 040_049.txt 042_041.txt 050_051.txt 054_052.txt 058_047.txt 059_048.txt 062_074.txt 063_060.txt 070_071.txt 073_064.txt 075_069.txt 076_066.txt 082_087.txt 085_083.txt 088_097.txt 092_086.txt 096_091.txt 099_081.txt 102_110.txt 105_100.txt 109_119.txt 111_113.txt 116_108.txt 118_106.txt 123_121.txt 126_132.txt 127_120.txt 130_125.txt 138_135.txt 139_122.txt 141_157.txt 143_154.txt 147_159.txt 148_145.txt 150_140.txt 153_144.txt 160_177.txt 162_168.txt 170_169.txt 172_179.txt 175_164.txt 176_163.txt 181_188.txt 185_192.txt 186_193.txt 190_189.txt 195_182.txt 197_194.txt"
    else:
        "Invalid option for opt.faust"
    
    for file in FILES.split():
        inf.forward(inputA=os.path.join(opt.dataset_path, 'test_scan_' + file[:3] + '.ply'),
                    inputB=os.path.join(opt.dataset_path, 'test_scan_' + file[4:7] + '.ply'),
                    path=file)     
    
    os.system(f"cd {opt.dir_name}; zip {opt.faust.lower()}.zip {FILES}; cd ..")

    

if __name__ == '__main__':
    import argument_parser

    opt = argument_parser.parser()
    my_utils.plant_seeds(randomized_seed=opt.randomize)

    import trainer

    trainer = trainer.Trainer(opt)
    trainer.build_network()
    main(opt, trainer.network)

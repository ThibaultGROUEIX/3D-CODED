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
    if opt.point_translation:
        network.make_high_res_template_from_low_res()
    inf = correspondences.Inference(model_path=opt.model_path, save_path=opt.dir_name, LR_input=opt.LR_input,
                                    network=network, HR=opt.HR, reg_num_steps=opt.reg_num_steps)

    if opt.faust == "INTER":
        inf.forward(inputA=os.path.join(opt.dataset_path, "test_scan_006.ply"),
                    inputB=os.path.join(opt.dataset_path, "test_scan_021.ply"), path="006_021.txt")
        inf.forward(inputA=os.path.join(opt.dataset_path, "test_scan_011.ply"),
                    inputB=os.path.join(opt.dataset_path, "test_scan_107.ply"), path="011_107.txt")
        inf.forward(inputA=os.path.join(opt.dataset_path, "test_scan_012.ply"),
                    inputB=os.path.join(opt.dataset_path, "test_scan_043.ply"), path="012_043.txt")
        inf.forward(inputA=os.path.join(opt.dataset_path, "test_scan_015.ply"),
                    inputB=os.path.join(opt.dataset_path, "test_scan_098.ply"), path="015_098.txt")
        inf.forward(inputA=os.path.join(opt.dataset_path, "test_scan_028.ply"),
                    inputB=os.path.join(opt.dataset_path, "test_scan_079.ply"), path="028_079.txt")
        inf.forward(inputA=os.path.join(opt.dataset_path, "test_scan_033.ply"),
                    inputB=os.path.join(opt.dataset_path, "test_scan_080.ply"), path="033_080.txt")
        inf.forward(inputA=os.path.join(opt.dataset_path, "test_scan_038.ply"),
                    inputB=os.path.join(opt.dataset_path, "test_scan_196.ply"), path="038_196.txt")
        inf.forward(inputA=os.path.join(opt.dataset_path, "test_scan_039.ply"),
                    inputB=os.path.join(opt.dataset_path, "test_scan_142.ply"), path="039_142.txt")
        inf.forward(inputA=os.path.join(opt.dataset_path, "test_scan_045.ply"),
                    inputB=os.path.join(opt.dataset_path, "test_scan_114.ply"), path="045_114.txt")
        inf.forward(inputA=os.path.join(opt.dataset_path, "test_scan_046.ply"),
                    inputB=os.path.join(opt.dataset_path, "test_scan_128.ply"), path="046_128.txt")
        inf.forward(inputA=os.path.join(opt.dataset_path, "test_scan_055.ply"),
                    inputB=os.path.join(opt.dataset_path, "test_scan_191.ply"), path="055_191.txt")
        inf.forward(inputA=os.path.join(opt.dataset_path, "test_scan_056.ply"),
                    inputB=os.path.join(opt.dataset_path, "test_scan_077.ply"), path="056_077.txt")
        inf.forward(inputA=os.path.join(opt.dataset_path, "test_scan_061.ply"),
                    inputB=os.path.join(opt.dataset_path, "test_scan_104.ply"), path="061_104.txt")
        inf.forward(inputA=os.path.join(opt.dataset_path, "test_scan_065.ply"),
                    inputB=os.path.join(opt.dataset_path, "test_scan_165.ply"), path="065_165.txt")
        inf.forward(inputA=os.path.join(opt.dataset_path, "test_scan_067.ply"),
                    inputB=os.path.join(opt.dataset_path, "test_scan_019.ply"), path="067_019.txt")
        inf.forward(inputA=os.path.join(opt.dataset_path, "test_scan_078.ply"),
                    inputB=os.path.join(opt.dataset_path, "test_scan_171.ply"), path="078_171.txt")
        inf.forward(inputA=os.path.join(opt.dataset_path, "test_scan_084.ply"),
                    inputB=os.path.join(opt.dataset_path, "test_scan_183.ply"), path="084_183.txt")
        inf.forward(inputA=os.path.join(opt.dataset_path, "test_scan_089.ply"),
                    inputB=os.path.join(opt.dataset_path, "test_scan_134.ply"), path="089_134.txt")
        inf.forward(inputA=os.path.join(opt.dataset_path, "test_scan_090.ply"),
                    inputB=os.path.join(opt.dataset_path, "test_scan_068.ply"), path="090_068.txt")
        inf.forward(inputA=os.path.join(opt.dataset_path, "test_scan_093.ply"),
                    inputB=os.path.join(opt.dataset_path, "test_scan_010.ply"), path="093_010.txt")
        inf.forward(inputA=os.path.join(opt.dataset_path, "test_scan_094.ply"),
                    inputB=os.path.join(opt.dataset_path, "test_scan_133.ply"), path="094_133.txt")
        inf.forward(inputA=os.path.join(opt.dataset_path, "test_scan_095.ply"),
                    inputB=os.path.join(opt.dataset_path, "test_scan_117.ply"), path="095_117.txt")
        inf.forward(inputA=os.path.join(opt.dataset_path, "test_scan_101.ply"),
                    inputB=os.path.join(opt.dataset_path, "test_scan_152.ply"), path="101_152.txt")
        inf.forward(inputA=os.path.join(opt.dataset_path, "test_scan_115.ply"),
                    inputB=os.path.join(opt.dataset_path, "test_scan_199.ply"), path="115_199.txt")
        inf.forward(inputA=os.path.join(opt.dataset_path, "test_scan_124.ply"),
                    inputB=os.path.join(opt.dataset_path, "test_scan_173.ply"), path="124_173.txt")
        inf.forward(inputA=os.path.join(opt.dataset_path, "test_scan_129.ply"),
                    inputB=os.path.join(opt.dataset_path, "test_scan_053.ply"), path="129_053.txt")
        inf.forward(inputA=os.path.join(opt.dataset_path, "test_scan_131.ply"),
                    inputB=os.path.join(opt.dataset_path, "test_scan_174.ply"), path="131_174.txt")
        inf.forward(inputA=os.path.join(opt.dataset_path, "test_scan_136.ply"),
                    inputB=os.path.join(opt.dataset_path, "test_scan_002.ply"), path="136_002.txt")
        inf.forward(inputA=os.path.join(opt.dataset_path, "test_scan_137.ply"),
                    inputB=os.path.join(opt.dataset_path, "test_scan_072.ply"), path="137_072.txt")
        inf.forward(inputA=os.path.join(opt.dataset_path, "test_scan_146.ply"),
                    inputB=os.path.join(opt.dataset_path, "test_scan_025.ply"), path="146_025.txt")
        inf.forward(inputA=os.path.join(opt.dataset_path, "test_scan_149.ply"),
                    inputB=os.path.join(opt.dataset_path, "test_scan_057.ply"), path="149_057.txt")
        inf.forward(inputA=os.path.join(opt.dataset_path, "test_scan_151.ply"),
                    inputB=os.path.join(opt.dataset_path, "test_scan_184.ply"), path="151_184.txt")
        inf.forward(inputA=os.path.join(opt.dataset_path, "test_scan_155.ply"),
                    inputB=os.path.join(opt.dataset_path, "test_scan_112.ply"), path="155_112.txt")
        inf.forward(inputA=os.path.join(opt.dataset_path, "test_scan_156.ply"),
                    inputB=os.path.join(opt.dataset_path, "test_scan_014.ply"), path="156_014.txt")
        inf.forward(inputA=os.path.join(opt.dataset_path, "test_scan_158.ply"),
                    inputB=os.path.join(opt.dataset_path, "test_scan_187.ply"), path="158_187.txt")
        inf.forward(inputA=os.path.join(opt.dataset_path, "test_scan_161.ply"),
                    inputB=os.path.join(opt.dataset_path, "test_scan_103.ply"), path="161_103.txt")
        inf.forward(inputA=os.path.join(opt.dataset_path, "test_scan_166.ply"),
                    inputB=os.path.join(opt.dataset_path, "test_scan_037.ply"), path="166_037.txt")
        inf.forward(inputA=os.path.join(opt.dataset_path, "test_scan_167.ply"),
                    inputB=os.path.join(opt.dataset_path, "test_scan_044.ply"), path="167_044.txt")
        inf.forward(inputA=os.path.join(opt.dataset_path, "test_scan_178.ply"),
                    inputB=os.path.join(opt.dataset_path, "test_scan_180.ply"), path="178_180.txt")
        inf.forward(inputA=os.path.join(opt.dataset_path, "test_scan_198.ply"),
                    inputB=os.path.join(opt.dataset_path, "test_scan_029.ply"), path="198_029.txt")

        FILES = " 006_021.txt 011_107.txt 012_043.txt 015_098.txt 028_079.txt 033_080.txt 038_196.txt 039_142.txt 045_114.txt 046_128.txt 055_191.txt 056_077.txt 061_104.txt 065_165.txt 067_019.txt 078_171.txt 084_183.txt 089_134.txt 090_068.txt 093_010.txt 094_133.txt 095_117.txt 101_152.txt 115_199.txt 124_173.txt 129_053.txt 131_174.txt 136_002.txt 137_072.txt 146_025.txt 149_057.txt 151_184.txt 155_112.txt 156_014.txt 158_187.txt 161_103.txt 166_037.txt 167_044.txt 178_180.txt 198_029.txt"
        os.system(f"cd {opt.dir_name}; zip inter.zip {FILES}")
        # os.system(f"cd {dir_name}; zip -r base{opt.id}.zip *_*.txt ; cd ../../; rm -r {dir_name}")

    elif opt.faust == "INTRA":
        inf.forward(inputA=os.path.join(opt.dataset_path, "test_scan_000.ply"),
                    inputB=os.path.join(opt.dataset_path, "test_scan_003.ply"), path="000_003.txt")
        inf.forward(inputA=os.path.join(opt.dataset_path, "test_scan_004.ply"),
                    inputB=os.path.join(opt.dataset_path, "test_scan_008.ply"), path="004_008.txt")
        inf.forward(inputA=os.path.join(opt.dataset_path, "test_scan_005.ply"),
                    inputB=os.path.join(opt.dataset_path, "test_scan_001.ply"), path="005_001.txt")
        inf.forward(inputA=os.path.join(opt.dataset_path, "test_scan_007.ply"),
                    inputB=os.path.join(opt.dataset_path, "test_scan_013.ply"), path="007_013.txt")
        inf.forward(inputA=os.path.join(opt.dataset_path, "test_scan_009.ply"),
                    inputB=os.path.join(opt.dataset_path, "test_scan_016.ply"), path="009_016.txt")
        inf.forward(inputA=os.path.join(opt.dataset_path, "test_scan_018.ply"),
                    inputB=os.path.join(opt.dataset_path, "test_scan_017.ply"), path="018_017.txt")
        inf.forward(inputA=os.path.join(opt.dataset_path, "test_scan_022.ply"),
                    inputB=os.path.join(opt.dataset_path, "test_scan_020.ply"), path="022_020.txt")
        inf.forward(inputA=os.path.join(opt.dataset_path, "test_scan_024.ply"),
                    inputB=os.path.join(opt.dataset_path, "test_scan_023.ply"), path="024_023.txt")
        inf.forward(inputA=os.path.join(opt.dataset_path, "test_scan_026.ply"),
                    inputB=os.path.join(opt.dataset_path, "test_scan_030.ply"), path="026_030.txt")
        inf.forward(inputA=os.path.join(opt.dataset_path, "test_scan_032.ply"),
                    inputB=os.path.join(opt.dataset_path, "test_scan_035.ply"), path="032_035.txt")
        inf.forward(inputA=os.path.join(opt.dataset_path, "test_scan_034.ply"),
                    inputB=os.path.join(opt.dataset_path, "test_scan_031.ply"), path="034_031.txt")
        inf.forward(inputA=os.path.join(opt.dataset_path, "test_scan_036.ply"),
                    inputB=os.path.join(opt.dataset_path, "test_scan_027.ply"), path="036_027.txt")
        inf.forward(inputA=os.path.join(opt.dataset_path, "test_scan_040.ply"),
                    inputB=os.path.join(opt.dataset_path, "test_scan_049.ply"), path="040_049.txt")
        inf.forward(inputA=os.path.join(opt.dataset_path, "test_scan_042.ply"),
                    inputB=os.path.join(opt.dataset_path, "test_scan_041.ply"), path="042_041.txt")
        inf.forward(inputA=os.path.join(opt.dataset_path, "test_scan_050.ply"),
                    inputB=os.path.join(opt.dataset_path, "test_scan_051.ply"), path="050_051.txt")
        inf.forward(inputA=os.path.join(opt.dataset_path, "test_scan_054.ply"),
                    inputB=os.path.join(opt.dataset_path, "test_scan_052.ply"), path="054_052.txt")
        inf.forward(inputA=os.path.join(opt.dataset_path, "test_scan_058.ply"),
                    inputB=os.path.join(opt.dataset_path, "test_scan_047.ply"), path="058_047.txt")
        inf.forward(inputA=os.path.join(opt.dataset_path, "test_scan_059.ply"),
                    inputB=os.path.join(opt.dataset_path, "test_scan_048.ply"), path="059_048.txt")
        inf.forward(inputA=os.path.join(opt.dataset_path, "test_scan_062.ply"),
                    inputB=os.path.join(opt.dataset_path, "test_scan_074.ply"), path="062_074.txt")
        inf.forward(inputA=os.path.join(opt.dataset_path, "test_scan_063.ply"),
                    inputB=os.path.join(opt.dataset_path, "test_scan_060.ply"), path="063_060.txt")
        inf.forward(inputA=os.path.join(opt.dataset_path, "test_scan_070.ply"),
                    inputB=os.path.join(opt.dataset_path, "test_scan_071.ply"), path="070_071.txt")
        inf.forward(inputA=os.path.join(opt.dataset_path, "test_scan_073.ply"),
                    inputB=os.path.join(opt.dataset_path, "test_scan_064.ply"), path="073_064.txt")
        inf.forward(inputA=os.path.join(opt.dataset_path, "test_scan_075.ply"),
                    inputB=os.path.join(opt.dataset_path, "test_scan_069.ply"), path="075_069.txt")
        inf.forward(inputA=os.path.join(opt.dataset_path, "test_scan_076.ply"),
                    inputB=os.path.join(opt.dataset_path, "test_scan_066.ply"), path="076_066.txt")
        inf.forward(inputA=os.path.join(opt.dataset_path, "test_scan_082.ply"),
                    inputB=os.path.join(opt.dataset_path, "test_scan_087.ply"), path="082_087.txt")
        inf.forward(inputA=os.path.join(opt.dataset_path, "test_scan_085.ply"),
                    inputB=os.path.join(opt.dataset_path, "test_scan_083.ply"), path="085_083.txt")
        inf.forward(inputA=os.path.join(opt.dataset_path, "test_scan_088.ply"),
                    inputB=os.path.join(opt.dataset_path, "test_scan_097.ply"), path="088_097.txt")
        inf.forward(inputA=os.path.join(opt.dataset_path, "test_scan_092.ply"),
                    inputB=os.path.join(opt.dataset_path, "test_scan_086.ply"), path="092_086.txt")
        inf.forward(inputA=os.path.join(opt.dataset_path, "test_scan_096.ply"),
                    inputB=os.path.join(opt.dataset_path, "test_scan_091.ply"), path="096_091.txt")
        inf.forward(inputA=os.path.join(opt.dataset_path, "test_scan_099.ply"),
                    inputB=os.path.join(opt.dataset_path, "test_scan_081.ply"), path="099_081.txt")
        inf.forward(inputA=os.path.join(opt.dataset_path, "test_scan_102.ply"),
                    inputB=os.path.join(opt.dataset_path, "test_scan_110.ply"), path="102_110.txt")
        inf.forward(inputA=os.path.join(opt.dataset_path, "test_scan_105.ply"),
                    inputB=os.path.join(opt.dataset_path, "test_scan_100.ply"), path="105_100.txt")
        inf.forward(inputA=os.path.join(opt.dataset_path, "test_scan_109.ply"),
                    inputB=os.path.join(opt.dataset_path, "test_scan_119.ply"), path="109_119.txt")
        inf.forward(inputA=os.path.join(opt.dataset_path, "test_scan_111.ply"),
                    inputB=os.path.join(opt.dataset_path, "test_scan_113.ply"), path="111_113.txt")
        inf.forward(inputA=os.path.join(opt.dataset_path, "test_scan_116.ply"),
                    inputB=os.path.join(opt.dataset_path, "test_scan_108.ply"), path="116_108.txt")
        inf.forward(inputA=os.path.join(opt.dataset_path, "test_scan_118.ply"),
                    inputB=os.path.join(opt.dataset_path, "test_scan_106.ply"), path="118_106.txt")
        inf.forward(inputA=os.path.join(opt.dataset_path, "test_scan_123.ply"),
                    inputB=os.path.join(opt.dataset_path, "test_scan_121.ply"), path="123_121.txt")
        inf.forward(inputA=os.path.join(opt.dataset_path, "test_scan_126.ply"),
                    inputB=os.path.join(opt.dataset_path, "test_scan_132.ply"), path="126_132.txt")
        inf.forward(inputA=os.path.join(opt.dataset_path, "test_scan_127.ply"),
                    inputB=os.path.join(opt.dataset_path, "test_scan_120.ply"), path="127_120.txt")
        inf.forward(inputA=os.path.join(opt.dataset_path, "test_scan_130.ply"),
                    inputB=os.path.join(opt.dataset_path, "test_scan_125.ply"), path="130_125.txt")
        inf.forward(inputA=os.path.join(opt.dataset_path, "test_scan_138.ply"),
                    inputB=os.path.join(opt.dataset_path, "test_scan_135.ply"), path="138_135.txt")
        inf.forward(inputA=os.path.join(opt.dataset_path, "test_scan_139.ply"),
                    inputB=os.path.join(opt.dataset_path, "test_scan_122.ply"), path="139_122.txt")
        inf.forward(inputA=os.path.join(opt.dataset_path, "test_scan_141.ply"),
                    inputB=os.path.join(opt.dataset_path, "test_scan_157.ply"), path="141_157.txt")
        inf.forward(inputA=os.path.join(opt.dataset_path, "test_scan_143.ply"),
                    inputB=os.path.join(opt.dataset_path, "test_scan_154.ply"), path="143_154.txt")
        inf.forward(inputA=os.path.join(opt.dataset_path, "test_scan_147.ply"),
                    inputB=os.path.join(opt.dataset_path, "test_scan_159.ply"), path="147_159.txt")
        inf.forward(inputA=os.path.join(opt.dataset_path, "test_scan_148.ply"),
                    inputB=os.path.join(opt.dataset_path, "test_scan_145.ply"), path="148_145.txt")
        inf.forward(inputA=os.path.join(opt.dataset_path, "test_scan_150.ply"),
                    inputB=os.path.join(opt.dataset_path, "test_scan_140.ply"), path="150_140.txt")
        inf.forward(inputA=os.path.join(opt.dataset_path, "test_scan_153.ply"),
                    inputB=os.path.join(opt.dataset_path, "test_scan_144.ply"), path="153_144.txt")
        inf.forward(inputA=os.path.join(opt.dataset_path, "test_scan_160.ply"),
                    inputB=os.path.join(opt.dataset_path, "test_scan_177.ply"), path="160_177.txt")
        inf.forward(inputA=os.path.join(opt.dataset_path, "test_scan_162.ply"),
                    inputB=os.path.join(opt.dataset_path, "test_scan_168.ply"), path="162_168.txt")
        inf.forward(inputA=os.path.join(opt.dataset_path, "test_scan_170.ply"),
                    inputB=os.path.join(opt.dataset_path, "test_scan_169.ply"), path="170_169.txt")
        inf.forward(inputA=os.path.join(opt.dataset_path, "test_scan_172.ply"),
                    inputB=os.path.join(opt.dataset_path, "test_scan_179.ply"), path="172_179.txt")
        inf.forward(inputA=os.path.join(opt.dataset_path, "test_scan_175.ply"),
                    inputB=os.path.join(opt.dataset_path, "test_scan_164.ply"), path="175_164.txt")
        inf.forward(inputA=os.path.join(opt.dataset_path, "test_scan_176.ply"),
                    inputB=os.path.join(opt.dataset_path, "test_scan_163.ply"), path="176_163.txt")
        inf.forward(inputA=os.path.join(opt.dataset_path, "test_scan_181.ply"),
                    inputB=os.path.join(opt.dataset_path, "test_scan_188.ply"), path="181_188.txt")
        inf.forward(inputA=os.path.join(opt.dataset_path, "test_scan_185.ply"),
                    inputB=os.path.join(opt.dataset_path, "test_scan_192.ply"), path="185_192.txt")
        inf.forward(inputA=os.path.join(opt.dataset_path, "test_scan_186.ply"),
                    inputB=os.path.join(opt.dataset_path, "test_scan_193.ply"), path="186_193.txt")
        inf.forward(inputA=os.path.join(opt.dataset_path, "test_scan_190.ply"),
                    inputB=os.path.join(opt.dataset_path, "test_scan_189.ply"), path="190_189.txt")
        inf.forward(inputA=os.path.join(opt.dataset_path, "test_scan_195.ply"),
                    inputB=os.path.join(opt.dataset_path, "test_scan_182.ply"), path="195_182.txt")
        inf.forward(inputA=os.path.join(opt.dataset_path, "test_scan_197.ply"),
                    inputB=os.path.join(opt.dataset_path, "test_scan_194.ply"), path="197_194.txt")

        FILES = "000_003.txt 004_008.txt 005_001.txt 007_013.txt 009_016.txt 018_017.txt 022_020.txt 024_023.txt 026_030.txt 032_035.txt 034_031.txt 036_027.txt 040_049.txt 042_041.txt 050_051.txt 054_052.txt 058_047.txt 059_048.txt 062_074.txt 063_060.txt 070_071.txt 073_064.txt 075_069.txt 076_066.txt 082_087.txt 085_083.txt 088_097.txt 092_086.txt 096_091.txt 099_081.txt 102_110.txt 105_100.txt 109_119.txt 111_113.txt 116_108.txt 118_106.txt 123_121.txt 126_132.txt 127_120.txt 130_125.txt 138_135.txt 139_122.txt 141_157.txt 143_154.txt 147_159.txt 148_145.txt 150_140.txt 153_144.txt 160_177.txt 162_168.txt 170_169.txt 172_179.txt 175_164.txt 176_163.txt 181_188.txt 185_192.txt 186_193.txt 190_189.txt 195_182.txt 197_194.txt"

        os.system(f"cd {opt.dir_name}; zip intra.zip {FILES}")
        # os.system(f"cd {dir_name}; zip -r base{opt.id}.zip *_*.txt ; cd ../../; rm -r {dir_name}")
    else:
        "Invalid option for opt.faust"


if __name__ == '__main__':
    import argument_parser

    opt = argument_parser.parser()
    my_utils.plant_seeds(randomized_seed=opt.randomize)

    import trainer

    trainer = trainer.Trainer(opt)
    trainer.build_network()
    main(opt, trainer.network)

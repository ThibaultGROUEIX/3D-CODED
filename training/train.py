import sys
sys.path.append("./auxiliary/")
sys.path.append("./inference/")
sys.path.append("./extension/")
sys.path.append('/app/python/')

import argument_parser
import my_utils
import script
import time

opt = argument_parser.parser()
my_utils.plant_seeds(randomized_seed=opt.randomize)

import trainer

trainer = trainer.Trainer(opt)
trainer.build_dataset_train()
trainer.build_dataset_test()
trainer.build_network()
trainer.build_optimizer()
trainer.build_losses()
trainer.start_train_time = time.time()

for epoch in range(opt.nepoch):
    trainer.train_epoch()
    trainer.test_epoch()
    trainer.dump_stats()
    trainer.save_network()
    trainer.increment_epoch()

trainer.save_new_experiments_results()
script.main(opt, trainer.network) #Inference
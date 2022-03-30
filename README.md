# Fashion-MNIST-part2

# PART2

### Objectives
* Write baseline code with Keras
* Use Tensorboard to visualize the loss, accuracy etc. 
* Use GitHub workflow structure
* Integrate CI/CD pipeline with Heroku
* Create folder structure for the ML project

### Code development
1. Create the train.py 
   1. Basically copy/paste the code from Fashion MNIST Jupyter notebook 
2. Create the config.py 
   1. To keep configuration settings for the training model  
3. Add function to save the training model into the train.py

### Tensorboard settings
1. Add tensorboard functionality to train.py 
2. Use git commit number to adjust the name of log files which will be saved by Tensorboard. 
3. After running the command "tensorboard --logdir logs/fit" in terminal, you can see the output on the localhost.  
4. Add validation data to model.fit function to see the visualization of validation in tensorboard as well. 

### GitHub Workflow
Use GitHub workflow to make different experiments on your model by creating a new branch. Each time you update it will automatically start building the system.
During this process, you can also observe the output of the system by adjusting the cml.yaml file. After the changes are done, this branch can be merged with main branch. 
1. Add visuzaliation(plots) to train.py for loss, accuracy 
2. Add "debug" parameter to config.py 
   1. Change debug parameter as True and False and observe the visualization(plots) 
3. Create necessary files to use GitHub workflow
   1. Add .github/workflows/cml.yaml file
   2. You can check the [CML Github page][1] for more detail
   3. You can also check the [CML example video][2]
4. Explore and write the cml.yaml file
   1. Add cml commands to see output 

### Update project structure
1. Add train_eval.py and define the necessary functions
2. Add main.py file and use the desired functions of train_eval.py

### Heroku Platform
Heroku platform is mainly used for web-based application however we can also use for any project. 
It mainly has deployement, stage and production steps. If your project is ready to deliver to the customer, it should be in production step. 

1. You can create new app in [Heroku][3] and Connect your GitHub account and deploy the branch 
2. You can create a pipeline with proper name and adjust your deliverable as "production" and "staging"
3. If the project is ready, then deliver it to customer by using "Promote to production" option.
4. Whenever you change the code, it will directly start building the pipeline. 
5. Check the example to use [Heroku example][4] for your development and production code. 

### Additional References
To build better models faster with experiment tracking, dataset versioning, and model management check [weights and biases][5]

[1]: https://github.com/iterative/cml
[2]: https://www.youtube.com/watch?v=9BgIDqAzfuA&list=PL7WG7YrwYcnDBDuCkFbcyjnZQrdskFsBz&index=2&ab_channel=DVCorg
[3]: https://www.heroku.com/
[4]: https://www.youtube.com/watch?v=_tiecDrW6yY&list=WL&index=7&ab_channel=JamesWard
[5]: https://wandb.ai/site

# https://docs.microsoft.com/en-us/azure/machine-learning/how-to-train-pytorch

import os
import shutil

from azureml.core.workspace import Workspace
from azureml.core import Experiment
from azureml.core import Environment
from azureml.core import ScriptRunConfig

from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.core.conda_dependencies import CondaDependencies


# train MNIST or Fashion-MNIST model
####################################################

mnist = True

if mnist:
  script_to_run = 'train_trades_mnist.py'
  expName = 'Train_TRADES_MNIST'
  modelName = 'trades_mnist'
else:
  script_to_run = 'train_trades_fashion_mnist.py'
  expName = 'Train_TRADES_Fashion_MNIST'
  modelName = 'trades_fashion_mnist'

####################################################

project_folder = './'
# os.makedirs(project_folder, exist_ok=True)
# shutil.copy('pytorch_train.py', project_folder)

####################################################

# Create compute target and workspace object

ws = Workspace.from_config()
# cluster_name = "TeslaK80compute" # Old compute cluster
cluster_name = "TeslaV100compute"

try:
    compute_target = ComputeTarget(workspace=ws, name=cluster_name)
    print('Found existing compute target')
except ComputeTargetException:
    print('Creating a new compute target...')
    compute_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_NC6', 
                                                           max_nodes=4)

    compute_target = ComputeTarget.create(ws, cluster_name, compute_config)

    compute_target.wait_for_completion(show_output=True, min_node_count=None, timeout_in_minutes=20)

####################################################

# Define enviornment

curated_env_name = 'AzureML-PyTorch-1.6-GPU'
pytorch_env = Environment.get(workspace=ws, name=curated_env_name)

conda_deps = ["python=3.6.2", "pip=20.2.4"]
pip_deps = [ "azureml-core==1.26.0",
                "azureml-defaults==1.26.0",
                "azureml-telemetry==1.26.0",
                "azureml-train-restclients-hyperdrive==1.26.0",
                "azureml-train-core==1.26.0",
                "cmake==3.18.2",
                "torch==1.6.0",
                "torchvision==0.5.0",
                "mkl==2018.0.3",
                "horovod==0.20.0",
                "tensorboard==1.14.0",
                "future==0.17.1",
                "joblib" ]

pytorch_env = Environment(name="myenv")

conda_dep = CondaDependencies()
for cd in conda_deps:
  conda_dep.add_conda_package(cd)
for pip in pip_deps:
  conda_dep.add_pip_package(pip)

pytorch_env.python.conda_dependencies=conda_dep

# "pytorch==1.4.0")
# conda_dep.add_conda_package("python==3.7")
# conda_dep.add_conda_package("torchvision")
# # conda_dep.add_conda_package("torchaudio")
# conda_dep.add_conda_package("cudatoolkit=10.2")
# conda_dep.add_conda_package("joblib")

# # Please add an explicit pip dependency.  I'm adding one for you, but still nagging you.
# conda_dep.add_conda_package("pip")

# Adds dependencies to PythonSection of myenv
# pytorch_env.python.conda_dependencies=conda_dep

####################################################

# Configure run

src = ScriptRunConfig(source_directory=project_folder,
                      script=script_to_run,
                    #   arguments=['--num_epochs', 30, '--output_dir', './outputs'],
                      compute_target=compute_target,
                      environment=pytorch_env)

####################################################

# Submit run

run = Experiment(ws, name=expName).submit(src)
run.wait_for_completion(show_output=True)

for f in run.get_file_names():
    print(f)

# Create model

# model = run.register_model(model_name='{}'.format(modelName), model_path='outputs/model.pt')
run.download_file(name='outputs/model.pt', output_file_path='model.pt')

# model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), '{}.pt'.format(modelName))
# model = run.register_model(model_name='pytorch-birds', model_path='model-mnist-smallCNN/{}.pt'.format(model))
# model = run.register_model(model_name='{}.pt'.format(model))
# model_dir = './model-mnist-smallCNN' # default model dir from train_trades_mnist.py
# model = run.register_model(model_name='pytorch-birds', model_path=os.path.join(model_dir, '{}.pt'.format(model)))

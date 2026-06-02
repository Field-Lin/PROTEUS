## Installation / 环境安装

### Prerequisites / 环境要求

Before installing PROTEUS, please ensure that one of the following Conda distributions is installed on your system:

在安装 PROTEUS 前，请确保您的系统已安装以下任意一种 Conda 发行版：

- Anaconda
- Miniconda

The project dependencies and Python version are managed through the provided `environment.yml` file.

项目所需的 Python 版本及全部依赖均已在 `environment.yml` 文件中配置。

------

### Create the Environment / 创建运行环境

Clone the repository and create the Conda environment using the provided configuration file:

克隆项目仓库，并根据提供的配置文件创建 Conda 环境：

```bash
conda env create -f environment.yml
```

After the installation is complete, activate the environment:

安装完成后，激活环境：

```bash
conda activate proteus
```

------

### Verify Installation / 验证安装

You can verify that the environment has been created successfully by running:

可以通过以下命令验证环境是否安装成功：

```bash
python --version
```

or

```bash
conda list
```

If no errors occur and all dependencies are listed correctly, the installation is successful.

若命令能够正常执行且依赖包显示正确，则说明安装成功。

------

### Update the Environment / 更新环境

If the project dependencies are updated in future releases, you can synchronize your local environment using:

如果项目后续更新了依赖配置，可以使用以下命令同步更新本地环境：

```bash
conda env update -f environment.yml --prune
```

The `--prune` option removes packages that are no longer required.

`--prune` 参数会自动移除当前环境中已不再需要的依赖包。
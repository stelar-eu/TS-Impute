## TS-Impute
The STELAR module for time series imputation.

### Overview
TS-Impute is a Python library providing the following functionalities: 
* Gap generation
* Imputation
* Train Ensemble Model
* Ensemble model Imputation

### Quick start
Please see the provided [notebooks](https://github.com/stelar-eu/TS-Impute/tree/main/notebooks).

### Installation
TS-Impute needs python version >=3.8 and < 4.0.

#### Python Module - Local library
TS-Impute, after it is downloaded from [here](https://github.com/stelar-eu/TS-Impute) can be installed with:
```sh
$ cd TS-Impute
$ chmod +x install_custom_library.sh
$ ./install_custom_library.sh
```
#### How to import local library
After you install TS-Impute as a local library you can import it in your python:

```python
import stelarImputation
```

### Docker Instructions
To build the image run:
```
docker build -t stelarimputation .
```

To create the container run:
```
docker run stelarimputation <token> <endpoint_url> <task_exec_id>
```

Please examine the JSON files in the [template_inputs](https://github.com/stelar-eu/TS-Impute/tree/main/template_inputs) directory to understand 
the required parameters for each functionality.

### License
The contents of this project are licensed under the [MIT License](https://github.com/stelar-eu/TS-Impute/blob/main/LICENSE).
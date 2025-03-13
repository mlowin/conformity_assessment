# Conformity assessments

This repository contains an implementation for (semi-)automated assessments of artificial intelligence (AI) conformity.

# Installation 
The implementation dependends on several other packages for handling data, modeling, and generating reports.
To install the required packages, run the following:
```
pip install -r requirements.txt
 ```

Besides, the tool needs a running MySQL (compatible) database. Please import the SQL dump from the file conformity_assessment.sql into your SQL database and add the database credentials into flask_app/server.py (lines 12 and following, e.g., db_username, db_password, ...).
# Usage

Conformity assessments can be run on the exemplary use case of fraud detection:

- ...
- ...
+ ...
+ ...
- ...


# Citation
Please consider citing us if you find this helpful for your work:
```
@article{vonZahn.2025,
  title={Navigating AI conformity: a design framework to assess fairness, explainability, and performance},
  author={von Zahn, Moritz and Zacharias, Jan and Lowin, Maximilian and Chen, Johannes and Hinz, Oliver},
  journal={Electronic Markets},
  volume={35},
  number={24},
  year={2025},
  publisher={Springer}
}
 ```

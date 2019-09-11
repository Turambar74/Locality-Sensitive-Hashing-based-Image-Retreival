# Locality-Sensitive-Hashing-based-Image-Retreival
Performs Locality-Sensitive Hashing on images after computing their feature vectors using a feature detection algorithm like BRIEF or SIFT. Stores the result in an LSHash object using Redis. When querying a new image, a feature detection algorithm is run on it and after performing a similarity search, the closest matching image from the dataset is returned.

# Setup Instructions
Install the lshash and future libraries. You may have to go to the directory the library is installed in and run 'python setup.py install'

Modify the path in the index.py and query.py files by replacing the predefined path in sys.path.insert with the path in which lshash is installed on your machine.

Install OpenCV

Follow the instructions in this link to install Redis : https://www.hackerearth.com/blog/python/getting-started-python-redis/
Then install redis-py

# Usage Instructions
Run index.py in the beginning and every time the image training set is updated. For the time being, execute flushall from redis-cli everytime before the lsh-indexing script is executed.

Run query.py each time image is to be queried.

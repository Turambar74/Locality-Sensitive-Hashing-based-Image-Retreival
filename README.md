# Locality-Sensitive-Hashing-based-Image-Retreival
Performs Locality-Sensitive Hashing on images after computing their feature vectors using a feature detection algorithm like BRIEF or SIFT. Stores the result in an LSHash object using Redis. When querying a new image, a feature detection algorithm is run on it and after performing a similarity search, the closest matching image from the dataset is returned.
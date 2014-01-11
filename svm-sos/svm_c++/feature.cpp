#include "feature.hpp"


std::vector<boost::shared_ptr<FeatureGroup>> GetFeaturesFromParam(STRUCT_LEARN_PARM* sparm) {
    std::vector<boost::shared_ptr<FeatureGroup>> features;
    features.push_back(boost::shared_ptr<FeatureGroup>(new GMMFeature(sparm->feature_scale)));
    if (sparm->distance_unary)
        features.push_back(boost::shared_ptr<FeatureGroup>(new DistanceFeature(sparm->feature_scale)));
    if (sparm->pairwise_feature) {
        std::cout << "Adding PairwiseFeature\n";
        features.push_back(boost::shared_ptr<FeatureGroup>(new PairwiseFeature(sparm->feature_scale)));
    }
    if (sparm->contrast_pairwise_feature) {
        std::cout << "Adding ContrastPairwiseFeature\n";
        features.push_back(boost::shared_ptr<FeatureGroup>(new ContrastPairwiseFeature(sparm->feature_scale)));
    }
    if (sparm->submodular_feature) {
        std::cout << "Adding SubmodularFeature\n";
        features.push_back(boost::shared_ptr<FeatureGroup>(new SubmodularFeature(sparm->feature_scale)));
    }
    return features;
}


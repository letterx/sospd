#include <cmath>
#include "svm_c++.hpp"
#include "image_manip.hpp"
#include "feature.hpp"
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/export.hpp>

PatternData::PatternData(const std::string& name, const cv::Mat& image, const cv::Mat& trimap) 
    : m_name(name), 
    m_image(image),
    m_bgdModel(),
    m_bgdGMM(m_bgdModel),
    m_fgdModel(),
    m_fgdGMM(m_fgdModel)
{
    ConvertToMask(trimap, m_tri);

    learnGMMs(m_image, m_tri, m_bgdGMM, m_fgdGMM);

    m_fgdUnaries.create(m_image.rows, m_image.cols, CV_64FC1);
    m_bgdUnaries.create(m_image.rows, m_image.cols, CV_64FC1);
    CalcUnaries(*this);

    m_beta = calcBeta(m_image);
    m_downW.create(m_image.rows, m_image.cols, CV_64FC1);
    m_rightW.create(m_image.rows, m_image.cols, CV_64FC1);
    std::function<void(const cv::Vec3b&, const cv::Vec3b&, double&, double&)> calcExpDiff = 
        [&](const cv::Vec3b& color1, const cv::Vec3b& color2, double& d1, double& d2) {
            cv::Vec3d c1 = color1;
            cv::Vec3d c2 = color2;
            cv::Vec3d diff = c1-c2;
            d1 = exp(-m_beta*diff.dot(diff));
            //d1 = abs(diff[0]) + abs(diff[1]) + abs(diff[2]);
    };
    ImageIterate(m_image, m_downW, cv::Point(0.0, 1.0), calcExpDiff);
    ImageIterate(m_image, m_rightW, cv::Point(1.0, 0.0), calcExpDiff);

}

LabelData::LabelData(const std::string& name, const cv::Mat& gt)
    : m_name(name)
{ 
    ConvertGreyToMask(gt, m_gt);
}

bool LabelData::operator==(const LabelData& l) const {
    ASSERT(m_gt.size() == l.m_gt.size());
    cv::MatConstIterator_<unsigned char> i1, e1, i2;
    i1 = m_gt.begin<unsigned char>();
    e1 = m_gt.end<unsigned char>();
    i2 = l.m_gt.begin<unsigned char>();
    for (; i1 != e1; ++i1, ++i2) {
        if (*i1 != *i2) {
            return false;
        }
    }
    return true;
}

double LabelData::Loss(const LabelData& l, double scale) const {
    ASSERT(m_gt.size() == l.m_gt.size());
    double loss = 0;
    ImageCIterate(m_gt, l.m_gt, 
            [&](const unsigned char& c1, const unsigned char& c2) {
                loss += LabelDiff(c1, c2);
            });
    loss /= (m_gt.rows*m_gt.cols);
    return loss*scale;
}


ModelData::ModelData() { }

void ModelData::InitFeatures(STRUCT_LEARN_PARM* sparm) {
    m_features = GetFeaturesFromParam(sparm);
}

long ModelData::NumFeatures() const {
    long n = 0;
    for (auto fgp : m_features) {
        n += fgp->NumFeatures();
    }
    return n;
}

void ModelData::InitializeCRF(CRF& crf, const PatternData& p) const {
    crf.AddNode(p.m_image.rows*p.m_image.cols);

}

void ModelData::AddLossToCRF(CRF& crf, const PatternData& p, const LabelData& l, double scale) const {
    double mult = scale/(p.m_image.rows*p.m_image.cols);
    cv::Point pt;
    for (pt.y = 0; pt.y < p.m_image.rows; ++pt.y) {
        for (pt.x = 0; pt.x < p.m_image.cols; ++pt.x) {
            CRF::NodeId id = pt.y * p.m_image.cols + pt.x;
            double E0 = 0;
            double E1 = 0;
            if (l.m_gt.at<unsigned char>(pt) == cv::GC_BGD) E1 -= 1.0*mult;
            if (l.m_gt.at<unsigned char>(pt) == cv::GC_FGD) E0 -= 1.0*mult;
            crf.AddUnaryTerm(id, doubleToREAL(E0), doubleToREAL(E1));
        }
    }
}

LabelData* ModelData::ExtractLabel(const CRF& crf, const PatternData& x) const {
    LabelData* lp = new LabelData;
    lp->m_name = x.m_name;
    lp->m_gt.create(x.m_image.rows, x.m_image.cols, CV_8UC1);
    CRF::NodeId id = 0;
    ImageIterate(lp->m_gt, 
        [&](unsigned char& c) { 
            //ASSERT(crf.GetLabel(id) >= 0);
            if (crf.GetLabel(id) == 0) c = cv::GC_BGD;
            else if (crf.GetLabel(id) == 1) c = cv::GC_FGD;
            else c = -1;
            id++;
        });
    //x.m_tri.copyTo(lp->m_gt);
    //cv::Mat bgdModel, fgdModel;
    //cv::grabCut(x.m_image, lp->m_gt, cv::Rect(), bgdModel, fgdModel, 10, cv::GC_INIT_WITH_MASK);
    return lp;
}

LabelData* ModelData::Classify(const PatternData& x, STRUCTMODEL* sm, STRUCT_LEARN_PARM* sparm) const {
    CRF crf;
    SubmodularFlow sf;
    HigherOrderWrapper ho;
    if (sparm->crf == 0) {
        crf.Wrap(&sf);
    } else {
        crf.Wrap(&ho);
    }
    InitializeCRF(crf, x);
    size_t feature_base = 1;
    for (auto fgp : m_features) {
        double violation = fgp->Violation(feature_base, sm->w);
        double w2 = 0;
        for (size_t i = feature_base; i < feature_base + fgp->NumFeatures(); ++i) 
            w2 += sm->w[i] * sm->w[i];
        if (violation > 0.0001 * w2) {
            std::cout << "*** Max Violation: " << violation << ", |w|^2: " << w2 << "***";
            std::cout.flush();
        }
        fgp->AddToCRF(crf, x, sm->w + feature_base );
        feature_base += fgp->NumFeatures();
    }
    crf.Solve();
    return ExtractLabel(crf, x);
}

LabelData* ModelData::FindMostViolatedConstraint(const PatternData& x, const LabelData& y, STRUCTMODEL* sm, STRUCT_LEARN_PARM* sparm) const {
    CRF crf;
    SubmodularFlow sf;
    HigherOrderWrapper ho;
    if (sparm->crf == 0) {
        crf.Wrap(&sf);
    } else {
        crf.Wrap(&ho);
    }
    InitializeCRF(crf, x);
    size_t feature_base = 1;
    for (auto fgp : m_features) {
        double violation = fgp->Violation(feature_base, sm->w);
        double w2 = 0;
        for (size_t i = feature_base; i < feature_base + fgp->NumFeatures(); ++i) 
            w2 += sm->w[i] * sm->w[i];
        if (violation > 0.0001 * w2) {
            std::cout << "*** Max Violation: " << violation << ", |w|^2: " << w2 << "***";
            std::cout.flush();
        }
        fgp->AddToCRF(crf, x, sm->w + feature_base );
        feature_base += fgp->NumFeatures();
    }
    AddLossToCRF(crf, x, y, sparm->loss_scale);
    crf.Solve();
    return ExtractLabel(crf, x);
}

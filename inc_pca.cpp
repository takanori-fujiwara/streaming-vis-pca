#include "inc_pca.hpp"

#include <iostream>

/* References
----------
T. Fujiwara, J.-K. Chou, Shilpika, P. Xu, L. Ren, K.-L. Ma. An Incremental
Dimensionality Reduction Method for Visualizing Streaming Multidimensional
Data
*/

IncPCA::IncPCA(Eigen::Index const nComponents, double const forgettingFactor)
    : nComponents(nComponents), forgettingFactor(forgettingFactor) {
  initialize();
}

void IncPCA::initialize() {
  components_.resize(0, 0);
  nSamplesSeen_ = 0;
  mean_ = 0.0;
  var_ = 0.0;
  singularValues_.resize(0, 0);
  explainedVariance_.resize(0, 0);
  explainedVarianceRatio_.resize(0, 0);
  noiseVariance_ = 0.0;
}

void IncPCA::partialFit(Eigen::MatrixXd X) {
  auto nSamples = X.rows();
  auto nFeatures = X.cols();

  // sanity check
  if (nComponents < 1) {
    nComponents_ = components_.rows() < 1 ? std::min(nSamples, nFeatures)
                                          : components_.rows();
  } else if (nComponents > nFeatures) {
    std::cerr << "nComponents must be smaller than or equal to nFeatures"
              << std::endl;
    return;
  } else if (nComponents > nSamples) {
    std::cerr << "nComponents must be smaller than or equal to nSamples"
              << std::endl;
  } else {
    nComponents_ = nComponents;
  }

  if (components_.rows() >= 1 && components_.rows() != nComponents_) {
    std::cerr << "size error";
  }

  // Initialize mean_ and var_
  if (nSamplesSeen_ == 0) {
    mean_ = Eigen::ArrayXd::Zero(nFeatures);
    var_ = Eigen::ArrayXd::Zero(nFeatures);
  }

  // Update stats
  auto colMean_colVar_nTotalSamples =
      incrementalMeanAndVar(X, forgettingFactor, mean_, var_, nSamplesSeen_);
  Eigen::RowVectorXd colMean = std::get<0>(colMean_colVar_nTotalSamples);
  Eigen::RowVectorXd colVar = std::get<1>(colMean_colVar_nTotalSamples);
  auto nTotalSamples = std::get<2>(colMean_colVar_nTotalSamples);

  // Whitening
  if (nSamplesSeen_ == 0) {
    // If it is the first step, simply whiten X
    X = X.rowwise() - colMean;
  } else {
    Eigen::RowVectorXd colBatchMean = X.colwise().mean();
    // If X has one row, we should not subtract mean (everything col becomes 0)
    if (X.rows() > 1) {
      X = X.rowwise() - colBatchMean;
    }
    Eigen::RowVectorXd castedMean_ = mean_;
    // Build matrix of combined previous basis and new data
    Eigen::RowVectorXd meanCorrection =
        std::sqrt((nSamplesSeen_ * nSamples) / nTotalSamples) *
        (castedMean_ - colBatchMean);

    singularValues_ = singularValues_.array() * forgettingFactor;
    Eigen::Map<Eigen::RowVectorXd> rvecSingularValues_(singularValues_.data(),
                                                       singularValues_.size());

    Eigen::MatrixXd sc = rvecSingularValues_ * components_;
    // vertical stack of result
    Eigen::MatrixXd tmpX(sc.rows() + X.rows() + meanCorrection.rows(),
                         sc.cols());
    tmpX << rvecSingularValues_ * components_, X, meanCorrection;
    X = tmpX;
  }

  // SVD decomposition
  Eigen::BDCSVD<Eigen::MatrixXd> svd(X,
                                     Eigen::ComputeThinU | Eigen::ComputeThinV);
  Eigen::MatrixXd U = svd.matrixU();
  Eigen::MatrixXd V =
      svd.matrixV().transpose(); // Eigen and numpy linalg svg has different
                                 // specification for return V
  Eigen::VectorXd S = svd.singularValues();

  svdFlip(U, V);

  Eigen::ArrayXd explainedVariance =
      pow(S.array(), 2.0) / double(nTotalSamples - 1);
  Eigen::ArrayXd explainedVarianceRatio =
      pow(S.array(), 2.0) / (colVar.array() * double(nTotalSamples)).sum();
  nSamplesSeen_ = Eigen::Index(forgettingFactor * nSamplesSeen_ +
                               (nTotalSamples - nSamplesSeen_));

  auto nRowsToTake = std::min(nComponents_, V.rows());
  components_ = V.block(0, 0, nRowsToTake, V.cols());
  singularValues_ = S.block(0, 0, nRowsToTake, S.cols());

  mean_ = colMean;
  var_ = colVar;
  explainedVariance_ =
      explainedVariance.block(0, 0, nRowsToTake, explainedVariance.cols());
  explainedVarianceRatio_ = explainedVarianceRatio.block(
      0, 0, nRowsToTake, explainedVarianceRatio.cols());
  if (nRowsToTake < nFeatures) {
    if (explainedVariance.rows() > nRowsToTake) {
      noiseVariance_ =
          explainedVariance
              .block(nRowsToTake, 0, explainedVariance.rows() - nRowsToTake,
                     explainedVariance.cols())
              .array()
              .mean();
    }
  } else {
    noiseVariance_ = 0.0;
  }
}

std::tuple<Eigen::ArrayXd, Eigen::ArrayXd, Eigen::Index>
IncPCA::incrementalMeanAndVar(Eigen::MatrixXd const &X, double forgettingFactor,
                              Eigen::ArrayXd const &lastMean,
                              Eigen::ArrayXd const &lastVariance,
                              Eigen::Index const lastSampleCount) {
  /* References
  ----------
  T. Chan, G. Golub, R. LeVeque. Algorithms for computing the sample
      variance: recommendations, The American Statistician, Vol. 37, No. 3,
      pp. 242-247
  */

  Eigen::ArrayXd lastSum =
      forgettingFactor * lastMean * double(lastSampleCount);
  Eigen::ArrayXd newSum = X.colwise().sum();
  auto newSampleCount = X.rows();
  auto updatedSampleCount =
      forgettingFactor * double(lastSampleCount) + double(newSampleCount);
  Eigen::ArrayXd updatedMean = (newSum + lastSum) / double(updatedSampleCount);

  Eigen::ArrayXd updatedVariance;
  Eigen::ArrayXd updatedUnnormalizedVariance;
  if (lastVariance.rows() != 0) {
    // colwise variace * newSampleCount
    Eigen::ArrayXd newUnnormalizedVariance =
        (pow(X.array(), 2.0).colwise().mean() -
         pow(X.colwise().mean().array(), 2.0)) *
        double(newSampleCount);

    if (lastSampleCount == 0) { // Avoid division by 0
      updatedUnnormalizedVariance = newUnnormalizedVariance;
    } else {
      auto lastOverNewCount = double(lastSampleCount) / double(newSampleCount);
      Eigen::ArrayXd lastUnnormalizedVariance =
          lastVariance * double(lastSampleCount);

      updatedUnnormalizedVariance =
          lastUnnormalizedVariance + newUnnormalizedVariance +
          lastOverNewCount / double(updatedSampleCount) *
              pow((lastSum / lastOverNewCount - newSum), 2.0);
    }
    updatedVariance = updatedUnnormalizedVariance / double(updatedSampleCount);
  }

  return std::make_tuple(updatedMean, updatedVariance,
                         lastSampleCount + newSampleCount);
}

void IncPCA::svdFlip(Eigen::MatrixXd &U, Eigen::MatrixXd &V) {
  auto n = U.cols();
  for (Eigen::Index i = 0, m = V.rows(); i < n; ++i) {
    Eigen::MatrixXd::Index maxIndex = 0;
    // Eigen maxCoeff doesn't work well for single cell matrix
    auto maxAbsPos =
        m == 1 ? 0 : Eigen::Index(V.row(i).array().abs().maxCoeff(&maxIndex));

    auto value = V(i, maxAbsPos);
    auto sign = (value > 0.0) - (value < 0.0);
    if (n > i)
      U.col(i) *= sign;
    V.row(i) *= sign;
  }
}

Eigen::MatrixXd IncPCA::transform(Eigen::MatrixXd const &X) {
  Eigen::MatrixXd result;
  if (components_.rows() > 0 && X.cols() == components_.cols()) {
    result = X.rowwise() - Eigen::RowVectorXd(mean_);
    result *= components_.transpose();
  } else {
    std::cerr << "Matrix cols and components cols have different sizes."
              << std::endl;
  }

  return result;
}

Eigen::MatrixXd IncPCA::getComponents() { return components_.transpose(); }

Eigen::MatrixXd IncPCA::getLoadings() {
  Eigen::MatrixXd result(components_.rows(), components_.cols());
  if (components_.rows() > 0) {
    Eigen::MatrixXd sqrtLambda = explainedVarianceRatio_.array().sqrt();
    for (Eigen::Index row = 0, m = components_.rows(); row < m; ++row) {
      for (Eigen::Index col = 0, n = components_.cols(); col < n; ++col) {
        result(row, col) = components_(row, col) * sqrtLambda(row);
      }
    }
  }
  return result;
}

double IncPCA::getUncertV(unsigned int const nObtainedFeatures) {
  auto result = 0.0;
  if (components_.rows() > 0) {
    auto k = std::min(components_.rows(), explainedVarianceRatio_.rows());
    auto sumR = explainedVarianceRatio_.block(0, 0, k, 1).array().sum();
    auto sumCoveredVal = 0.0;
    auto nFeatures = components_.cols();

    for (Eigen::Index i = 0; i < k; ++i) {
      Eigen::MatrixXd w = components_.block(i, 0, 1, nFeatures).array() *
                          std::sqrt(singularValues_(i, 0));

      auto a = w.block(0, 0, 1, nObtainedFeatures).array().abs().sum();
      auto b = w.array().abs().sum();
      b = std::max(b, 0.000000000000001); // to avoid zero div
      sumCoveredVal += a / b;
    }
    sumR = std::max(sumR, 0.000000000000001); // to avoid zero div
    result = sumCoveredVal / double(k);
  }
  return 1.0 - result;
}

Eigen::MatrixXd IncPCA::geomTrans(Eigen::MatrixXd const &pointsFrom,
                                  Eigen::MatrixXd const &pointsTo) {
  auto n = std::min(pointsFrom.rows(), pointsTo.rows());
  auto m = pointsTo.rows() - pointsFrom.rows();

  if (m < 0) {
    std::cerr << "geomTrans: does not support the case that rows of pointsTo "
                 "is smaller than rows of pointsFrom"
              << std::endl;
  }

  Eigen::MatrixXd processedPointsFrom = pointsFrom.topRows(n);
  Eigen::MatrixXd processedPointsTo = pointsTo.topRows(n);
  // translation
  Eigen::RowVectorXd meanPointsFrom = processedPointsFrom.colwise().mean();
  Eigen::RowVectorXd meanPointsTo = processedPointsTo.colwise().mean();
  processedPointsFrom = processedPointsFrom.rowwise() - meanPointsFrom;
  processedPointsTo = processedPointsTo.rowwise() - meanPointsTo;

  // uniform scaling
  double scalePointsFrom = processedPointsFrom.colwise().norm().sum();
  double scalePointsTo = processedPointsTo.colwise().norm().sum();
  scalePointsFrom /= double(n);
  scalePointsTo /= double(n);
  scalePointsFrom = std::sqrt(scalePointsFrom);
  scalePointsTo = std::sqrt(scalePointsTo);
  processedPointsFrom /= scalePointsFrom;
  processedPointsTo /= scalePointsTo;

  // rotation
  Eigen::BDCSVD<Eigen::MatrixXd> svd(processedPointsFrom.transpose() *
                                         processedPointsTo,
                                     Eigen::ComputeThinU | Eigen::ComputeThinV);
  Eigen::MatrixXd U = svd.matrixU();
  Eigen::MatrixXd V = svd.matrixV(); // before transposed unlike numpy svd
  Eigen::VectorXd S = svd.singularValues();
  Eigen::MatrixXd R = V * U.transpose();

  // apply tranformation
  Eigen::MatrixXd transformedPointsTo(pointsTo.rows(), 2);
  transformedPointsTo.topRows(n) = scalePointsFrom * processedPointsTo * R;

  // for new points
  Eigen::MatrixXd processedNewPointsTo = pointsTo.bottomRows(m);
  processedNewPointsTo = processedNewPointsTo.rowwise() - meanPointsTo;
  processedNewPointsTo /= scalePointsTo;
  processedNewPointsTo = scalePointsFrom * processedNewPointsTo * R;

  // aggregate result and move center
  transformedPointsTo.bottomRows(m) = processedNewPointsTo;
  transformedPointsTo = transformedPointsTo.rowwise() + meanPointsFrom;

  return transformedPointsTo;
}

void IncPCA::updateWithAdadelta(Eigen::Vector3d &theta,
                                Eigen::Vector3d &meanSqGrad,
                                Eigen::Vector3d &meanSqDTheta,
                                Eigen::Vector3d const &grad, double const rho,
                                double const epsilon) {
  meanSqGrad = rho * meanSqGrad.array() + (1.0 - rho) * grad.array().square();
  Eigen::Vector3d const dTheta = -((meanSqDTheta.array() + epsilon).sqrt() /
                                   (meanSqGrad.array() + epsilon).sqrt()) *
                                 grad.array();
  meanSqDTheta =
      rho * meanSqDTheta.array() + (1.0 - rho) * dTheta.array().square();

  theta = theta.array() + dTheta.array();
}

std::pair<Eigen::RowVector2d, double>
IncPCA::posEst(Eigen::RowVector2d const &posInPointsFrom,
               Eigen::MatrixX2d const &pointsFrom,
               Eigen::MatrixX2d const &pointsTo) {
  // calc distances
  Eigen::VectorXd dists =
      (pointsFrom.rowwise() - posInPointsFrom).rowwise().norm();

  // initialize position with centroid of pointsTo
  Eigen::RowVector2d initPos = pointsTo.colwise().mean();

  Eigen::Vector3d theta;
  theta << initPos(0), initPos(1), 1.0; // 1.0 is initial lambda
  Eigen::Vector3d estimatedTheta = optPosEst(theta, dists, pointsTo);
  double uncertU = calcUncertU(theta, dists, pointsTo);

  Eigen::RowVector2d pos;
  pos << estimatedTheta(0), estimatedTheta(1);

  return {pos, uncertU};
}

Eigen::Vector3d IncPCA::optPosEst(Eigen::Vector3d const &initTheta,
                                  Eigen::VectorXd const &distsInPointsFrom,
                                  Eigen::MatrixX2d const &pointsTo,
                                  size_t const iterations, double const rho,
                                  double const epsilon) {

  Eigen::Vector3d theta = initTheta;

  Eigen::Vector3d meanSqGrad = Eigen::Vector3d::Zero(initTheta.rows());
  Eigen::Vector3d meanSqDTheta = Eigen::Vector3d::Zero(initTheta.rows());

  auto n = std::min(distsInPointsFrom.size(), pointsTo.rows());

  for (size_t i = 0; i < iterations; ++i) {
    auto grad = calcGradPosEst(theta, distsInPointsFrom, pointsTo, n);
    updateWithAdadelta(theta, meanSqGrad, meanSqDTheta, grad, rho, epsilon);
  }

  return theta;
}

Eigen::Vector3d IncPCA::calcGradPosEst(Eigen::Vector3d const &theta,
                                       Eigen::VectorXd const &distsInPointsFrom,
                                       Eigen::MatrixX2d const &pointsTo,
                                       size_t const n) {
  /// sigmoid ver
  Eigen::RowVector2d pos;
  pos << theta(0), theta(1);
  auto z = theta(2); // zeta for sigmoid function
  auto g = 0.1;      // gain for sigmoid function
  auto s = 1.0 / (1.0 + std::exp(-g * z));
  auto c = 0.1; // capacity allows scaling betwen 1-c to 1+C
  auto b = 1.0 - c + 2.0 * c * s;
  auto c2gs1_s = 2.0 * c * g * s * (1.0 - s);

  Eigen::MatrixX2d dPos = (-1.0 * pointsTo).rowwise() + pos;
  Eigen::VectorXd bl2Norm = b * dPos.rowwise().norm();
  bl2Norm = bl2Norm.unaryExpr([](double v) { return v > 1e-6 ? v : 1e-6; });
  Eigen::VectorXd dist2_bl2Norm_1 =
      2.0 * (distsInPointsFrom.array() / bl2Norm.array() - 1.0);

  Eigen::ArrayXd gradXY =
      ((-b * dPos).array().colwise() * dist2_bl2Norm_1.array()).colwise().sum();
  double gradZ =
      ((-c2gs1_s * dist2_bl2Norm_1).array() * bl2Norm.array() * bl2Norm.array())
          .sum();

  Eigen::Vector3d result;
  result << gradXY(0), gradXY(1), gradZ;

  return result;
}

double IncPCA::calcUncertU(Eigen::Vector3d const &theta,
                           Eigen::VectorXd const &distsInPointsFrom,
                           Eigen::MatrixX2d const &pointsTo) {
  double result = 0.0;
  double err = errorPosEst(theta, distsInPointsFrom, pointsTo);
  double sumSqDistsInPointsFrom = distsInPointsFrom.array().square().sum();

  if (sumSqDistsInPointsFrom > 0.0) {
    result = std::sqrt(err / sumSqDistsInPointsFrom);
  }
  if (err > sumSqDistsInPointsFrom) {
    // sometimes \beta * s'_ui is larger than s_ui. If so, ucert should be 1.0
    result = 1.0;
  }

  return result;
}

double IncPCA::errorPosEst(Eigen::Vector3d const &theta,
                           Eigen::VectorXd const &distsInPointsFrom,
                           Eigen::MatrixX2d const &pointsTo) {
  /// sigmoid ver
  Eigen::RowVector2d pos;
  pos << theta(0), theta(1);
  auto z = theta(2); // zeta for sigmoid function
  auto g = 0.1;      // gain for sigmoid function
  auto s = 1.0 / (1.0 + std::exp(-g * z));
  auto c = 0.1; // capacity allows scaling betwen 1-c to 1+C
  auto b = 1.0 - c + 2.0 * c * s;

  Eigen::MatrixX2d dPos = (-1.0 * pointsTo).rowwise() + pos;
  Eigen::VectorXd bl2Norm = b * dPos.rowwise().norm();

  return ((distsInPointsFrom - bl2Norm).array().square()).sum();
}

std::tuple<double, double, double> IncPCA::updateUncertWeight(
    double currentGamma, double currentSqGrad, double currentSqDGamma,
    Eigen::MatrixXd const &sigma, std::vector<Eigen::MatrixXd> const &sprimes,
    Eigen::MatrixXd const &uncertUs, Eigen::VectorXd const &uncertVs) {
  // sigma: shape(m, n)
  // sprimes: D x shape(m, n)
  // uncrertUs: shape(D, m)
  // uncrertVs: shape(D)

  Eigen::Index m = sigma.rows();                 // # of new data points
  Eigen::Index n = sigma.cols();                 // # of exisiting data points
  Eigen::Index D = Eigen::Index(sprimes.size()); // # of full dimensions

  // check matrix sizes
  for (auto const &sprime : sprimes) {
    if (sprime.rows() != m || sprime.cols() != n) {
      std::cerr << "Sigma and one of Sprimes have different matrix sizes"
                << std::endl;
    }
  }
  if (uncertUs.rows() != D || uncertUs.cols() != m) {
    std::cerr << "uncertUs does not have D rows and m cols" << std::endl;
  }
  if (uncertVs.size() != D) {
    std::cerr << "uncertVs does not have D length" << std::endl;
  }

  Eigen::MatrixXd errs(D, m);
  for (Eigen::Index l = 0; l < D; ++l) {
    // mean absolute error
    Eigen::RowVectorXd mae(m);
    mae = (sigma - sprimes[l]).array().abs().rowwise().mean();
    errs.row(l) = mae;
  }

  Eigen::RowVectorXd errD = errs.row(D - 1);
  Eigen::RowVectorXd meanU = uncertUs.colwise().mean();
  // added abs in case error with D dims > error with l dims (l <= D)
  Eigen::RowVectorXd sumDiffErr =
      (errs.rowwise() - errD).array().abs().colwise().sum();
  double sumUncertV = uncertVs.sum();

  // to avoid zero div
  meanU = meanU.unaryExpr([](double v) { return v > 1e-6 ? v : 1e-6; });
  sumUncertV = std::max(sumUncertV, 1e-6);

  Eigen::RowVectorXd rho = errD.array() / meanU.array();
  Eigen::RowVectorXd phi = sumDiffErr / sumUncertV;

  // Altough using a different gamma for each dim is possible,
  // this implementation using average of rho and phi for simplification
  double meanRho = rho.mean();
  double meanPhi = phi.mean();

  if (meanRho + meanPhi > 0) {
    // update with adadelta
    double r = 0.95;
    double e = 1e-6;

    double grad = currentGamma - meanRho / (meanRho + meanPhi);
    currentSqGrad = r * currentSqGrad + (1.0 - r) * grad * grad;
    double dGamma =
        -grad * std::sqrt(currentSqDGamma + e) / std::sqrt(currentSqGrad + e);

    currentSqDGamma = r * currentSqDGamma + (1.0 - r) * dGamma * dGamma;

    currentGamma = currentGamma + dGamma;
  }

  return std::make_tuple(currentGamma, currentSqGrad, currentSqDGamma);
}

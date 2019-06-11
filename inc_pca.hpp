#ifndef IncPCA_HPP
#define IncPCA_HPP

#include "Eigen/Core"
#include "Eigen/Dense"
#include "Eigen/LU"
#include <array>
#include <tuple>
#include <vector>

class IncPCA {
public:
  IncPCA(Eigen::Index const nComponents = 2,
         double const forgettingFactor = 1.0);

  void initialize();
  void partialFit(Eigen::MatrixXd X);
  Eigen::MatrixXd transform(Eigen::MatrixXd const &X);
  Eigen::MatrixXd getComponents();
  Eigen::MatrixXd getLoadings();
  double getUncertV(unsigned int const nObtainedFeatures);

  Eigen::MatrixXd static geomTrans(Eigen::MatrixXd const &pointsFrom,
                                   Eigen::MatrixXd const &pointsTo);

  std::pair<Eigen::RowVector2d, double> static posEst(
      Eigen::RowVector2d const &posInPointsFrom,
      Eigen::MatrixX2d const &pointsFrom, Eigen::MatrixX2d const &pointsTo);

  std::tuple<double, double, double> static updateUncertWeight(
      double currentGamma, double currentSqGrad, double currentSqDGamma,
      Eigen::MatrixXd const &sigma, std::vector<Eigen::MatrixXd> const &sprimes,
      Eigen::MatrixXd const &uncertUs, Eigen::VectorXd const &uncertVs);

private:
  Eigen::Index nComponents;
  double forgettingFactor;

  Eigen::MatrixXd components_;
  Eigen::Index nComponents_;
  Eigen::Index nSamplesSeen_;
  Eigen::ArrayXd mean_;
  Eigen::ArrayXd var_;
  Eigen::MatrixXd singularValues_;
  Eigen::MatrixXd explainedVariance_;
  Eigen::MatrixXd explainedVarianceRatio_;
  double noiseVariance_;

  std::tuple<
      Eigen::ArrayXd, Eigen::ArrayXd,
      Eigen::Index> static incrementalMeanAndVar(Eigen::MatrixXd const &X,
                                                 double forgettingFactor,
                                                 Eigen::ArrayXd const
                                                     &lastMean = {},
                                                 Eigen::ArrayXd const
                                                     &lastVariance = {},
                                                 Eigen::Index const
                                                     lastSampleCount = 0);
  void static svdFlip(Eigen::MatrixXd &U, Eigen::MatrixXd &V);

  Eigen::Vector3d static optPosEst(Eigen::Vector3d const &initTheta,
                                   Eigen::VectorXd const &distsInPointsFrom,
                                   Eigen::MatrixX2d const &pointsTo,
                                   size_t const iterations = 1000,
                                   double const rho = 0.95,
                                   double const epsilon = 1e-6);
  Eigen::Vector3d static calcGradPosEst(
      Eigen::Vector3d const &theta, Eigen::VectorXd const &distsInPointsFrom,
      Eigen::MatrixX2d const &pointsTo, size_t const n);
  void static updateWithAdadelta(Eigen::Vector3d &theta,
                                 Eigen::Vector3d &meanSqGrad,
                                 Eigen::Vector3d &meanSqDTheta,
                                 Eigen::Vector3d const &grad, double const rho,
                                 double const epsilon);
  double static calcUncertU(Eigen::Vector3d const &theta,
                            Eigen::VectorXd const &distsInPointsFrom,
                            Eigen::MatrixX2d const &pointsTo);
  double static errorPosEst(Eigen::Vector3d const &theta,
                            Eigen::VectorXd const &distsInPointsFrom,
                            Eigen::MatrixX2d const &pointsTo);
};

#endif // IncPCA_HPP

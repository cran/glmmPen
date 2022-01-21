// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <RcppEigen.h>
#include <Rcpp.h>

using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

// grp_CD_XZ_fast
arma::vec grp_CD_XZ_fast(const arma::vec& y, const arma::mat& X, const arma::mat& Z, const arma::vec& group, SEXP pBigMat, const arma::sp_mat& J_q, arma::vec dims, arma::vec beta, const arma::vec& offset, const char* family, int link, int init, double phi, const arma::uvec& XZ_group, arma::uvec K, const char* penalty, arma::vec params, int trace);
RcppExport SEXP _glmmPen_grp_CD_XZ_fast(SEXP ySEXP, SEXP XSEXP, SEXP ZSEXP, SEXP groupSEXP, SEXP pBigMatSEXP, SEXP J_qSEXP, SEXP dimsSEXP, SEXP betaSEXP, SEXP offsetSEXP, SEXP familySEXP, SEXP linkSEXP, SEXP initSEXP, SEXP phiSEXP, SEXP XZ_groupSEXP, SEXP KSEXP, SEXP penaltySEXP, SEXP paramsSEXP, SEXP traceSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::vec& >::type y(ySEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type Z(ZSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type group(groupSEXP);
    Rcpp::traits::input_parameter< SEXP >::type pBigMat(pBigMatSEXP);
    Rcpp::traits::input_parameter< const arma::sp_mat& >::type J_q(J_qSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type dims(dimsSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type beta(betaSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type offset(offsetSEXP);
    Rcpp::traits::input_parameter< const char* >::type family(familySEXP);
    Rcpp::traits::input_parameter< int >::type link(linkSEXP);
    Rcpp::traits::input_parameter< int >::type init(initSEXP);
    Rcpp::traits::input_parameter< double >::type phi(phiSEXP);
    Rcpp::traits::input_parameter< const arma::uvec& >::type XZ_group(XZ_groupSEXP);
    Rcpp::traits::input_parameter< arma::uvec >::type K(KSEXP);
    Rcpp::traits::input_parameter< const char* >::type penalty(penaltySEXP);
    Rcpp::traits::input_parameter< arma::vec >::type params(paramsSEXP);
    Rcpp::traits::input_parameter< int >::type trace(traceSEXP);
    rcpp_result_gen = Rcpp::wrap(grp_CD_XZ_fast(y, X, Z, group, pBigMat, J_q, dims, beta, offset, family, link, init, phi, XZ_group, K, penalty, params, trace));
    return rcpp_result_gen;
END_RCPP
}
// soft_thresh
double soft_thresh(double zeta, double lambda);
RcppExport SEXP _glmmPen_soft_thresh(SEXP zetaSEXP, SEXP lambdaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< double >::type zeta(zetaSEXP);
    Rcpp::traits::input_parameter< double >::type lambda(lambdaSEXP);
    rcpp_result_gen = Rcpp::wrap(soft_thresh(zeta, lambda));
    return rcpp_result_gen;
END_RCPP
}
// MCP_soln
double MCP_soln(double zeta, double nu, double lambda, double gamma, double alpha);
RcppExport SEXP _glmmPen_MCP_soln(SEXP zetaSEXP, SEXP nuSEXP, SEXP lambdaSEXP, SEXP gammaSEXP, SEXP alphaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< double >::type zeta(zetaSEXP);
    Rcpp::traits::input_parameter< double >::type nu(nuSEXP);
    Rcpp::traits::input_parameter< double >::type lambda(lambdaSEXP);
    Rcpp::traits::input_parameter< double >::type gamma(gammaSEXP);
    Rcpp::traits::input_parameter< double >::type alpha(alphaSEXP);
    rcpp_result_gen = Rcpp::wrap(MCP_soln(zeta, nu, lambda, gamma, alpha));
    return rcpp_result_gen;
END_RCPP
}
// SCAD_soln
double SCAD_soln(double zeta, double nu, double lambda, double gamma, double alpha);
RcppExport SEXP _glmmPen_SCAD_soln(SEXP zetaSEXP, SEXP nuSEXP, SEXP lambdaSEXP, SEXP gammaSEXP, SEXP alphaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< double >::type zeta(zetaSEXP);
    Rcpp::traits::input_parameter< double >::type nu(nuSEXP);
    Rcpp::traits::input_parameter< double >::type lambda(lambdaSEXP);
    Rcpp::traits::input_parameter< double >::type gamma(gammaSEXP);
    Rcpp::traits::input_parameter< double >::type alpha(alphaSEXP);
    rcpp_result_gen = Rcpp::wrap(SCAD_soln(zeta, nu, lambda, gamma, alpha));
    return rcpp_result_gen;
END_RCPP
}
// pglm_fit
arma::vec pglm_fit(arma::vec y, arma::mat X, arma::vec dims, arma::vec beta, arma::vec offset, const char* family, int link, const char* penalty, arma::vec params, arma::vec penalty_factor, int trace);
RcppExport SEXP _glmmPen_pglm_fit(SEXP ySEXP, SEXP XSEXP, SEXP dimsSEXP, SEXP betaSEXP, SEXP offsetSEXP, SEXP familySEXP, SEXP linkSEXP, SEXP penaltySEXP, SEXP paramsSEXP, SEXP penalty_factorSEXP, SEXP traceSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type y(ySEXP);
    Rcpp::traits::input_parameter< arma::mat >::type X(XSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type dims(dimsSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type beta(betaSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type offset(offsetSEXP);
    Rcpp::traits::input_parameter< const char* >::type family(familySEXP);
    Rcpp::traits::input_parameter< int >::type link(linkSEXP);
    Rcpp::traits::input_parameter< const char* >::type penalty(penaltySEXP);
    Rcpp::traits::input_parameter< arma::vec >::type params(paramsSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type penalty_factor(penalty_factorSEXP);
    Rcpp::traits::input_parameter< int >::type trace(traceSEXP);
    rcpp_result_gen = Rcpp::wrap(pglm_fit(y, X, dims, beta, offset, family, link, penalty, params, penalty_factor, trace));
    return rcpp_result_gen;
END_RCPP
}
// sample_mc_inner_gibbs
List sample_mc_inner_gibbs(arma::mat f, arma::mat z, arma::vec y, arma::vec t, int NMC, arma::vec u0, const char* family, int link, double phi, double sig_g);
RcppExport SEXP _glmmPen_sample_mc_inner_gibbs(SEXP fSEXP, SEXP zSEXP, SEXP ySEXP, SEXP tSEXP, SEXP NMCSEXP, SEXP u0SEXP, SEXP familySEXP, SEXP linkSEXP, SEXP phiSEXP, SEXP sig_gSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type f(fSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type z(zSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type y(ySEXP);
    Rcpp::traits::input_parameter< arma::vec >::type t(tSEXP);
    Rcpp::traits::input_parameter< int >::type NMC(NMCSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type u0(u0SEXP);
    Rcpp::traits::input_parameter< const char* >::type family(familySEXP);
    Rcpp::traits::input_parameter< int >::type link(linkSEXP);
    Rcpp::traits::input_parameter< double >::type phi(phiSEXP);
    Rcpp::traits::input_parameter< double >::type sig_g(sig_gSEXP);
    rcpp_result_gen = Rcpp::wrap(sample_mc_inner_gibbs(f, z, y, t, NMC, u0, family, link, phi, sig_g));
    return rcpp_result_gen;
END_RCPP
}
// sample_mc_gibbs_adapt_rw
arma::mat sample_mc_gibbs_adapt_rw(arma::mat f, arma::mat z, arma::vec y, int NMC, arma::vec u0, arma::rowvec proposal_SD, int batch, int batch_length, int offset, int nMC_burnin, const char* family, int link, double phi, double sig_g);
RcppExport SEXP _glmmPen_sample_mc_gibbs_adapt_rw(SEXP fSEXP, SEXP zSEXP, SEXP ySEXP, SEXP NMCSEXP, SEXP u0SEXP, SEXP proposal_SDSEXP, SEXP batchSEXP, SEXP batch_lengthSEXP, SEXP offsetSEXP, SEXP nMC_burninSEXP, SEXP familySEXP, SEXP linkSEXP, SEXP phiSEXP, SEXP sig_gSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type f(fSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type z(zSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type y(ySEXP);
    Rcpp::traits::input_parameter< int >::type NMC(NMCSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type u0(u0SEXP);
    Rcpp::traits::input_parameter< arma::rowvec >::type proposal_SD(proposal_SDSEXP);
    Rcpp::traits::input_parameter< int >::type batch(batchSEXP);
    Rcpp::traits::input_parameter< int >::type batch_length(batch_lengthSEXP);
    Rcpp::traits::input_parameter< int >::type offset(offsetSEXP);
    Rcpp::traits::input_parameter< int >::type nMC_burnin(nMC_burninSEXP);
    Rcpp::traits::input_parameter< const char* >::type family(familySEXP);
    Rcpp::traits::input_parameter< int >::type link(linkSEXP);
    Rcpp::traits::input_parameter< double >::type phi(phiSEXP);
    Rcpp::traits::input_parameter< double >::type sig_g(sig_gSEXP);
    rcpp_result_gen = Rcpp::wrap(sample_mc_gibbs_adapt_rw(f, z, y, NMC, u0, proposal_SD, batch, batch_length, offset, nMC_burnin, family, link, phi, sig_g));
    return rcpp_result_gen;
END_RCPP
}
// invlink
arma::vec invlink(int link, arma::vec eta);
RcppExport SEXP _glmmPen_invlink(SEXP linkSEXP, SEXP etaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< int >::type link(linkSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type eta(etaSEXP);
    rcpp_result_gen = Rcpp::wrap(invlink(link, eta));
    return rcpp_result_gen;
END_RCPP
}
// Qfun
double Qfun(const arma::vec& y, const arma::mat& X, const arma::mat& Z, SEXP pBigMat, const arma::vec& group, const arma::sp_mat& J_q, const arma::vec& beta, const arma::vec offset, arma::vec dims, const char* family, int link, double sig_g, double phi);
RcppExport SEXP _glmmPen_Qfun(SEXP ySEXP, SEXP XSEXP, SEXP ZSEXP, SEXP pBigMatSEXP, SEXP groupSEXP, SEXP J_qSEXP, SEXP betaSEXP, SEXP offsetSEXP, SEXP dimsSEXP, SEXP familySEXP, SEXP linkSEXP, SEXP sig_gSEXP, SEXP phiSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::vec& >::type y(ySEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type Z(ZSEXP);
    Rcpp::traits::input_parameter< SEXP >::type pBigMat(pBigMatSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type group(groupSEXP);
    Rcpp::traits::input_parameter< const arma::sp_mat& >::type J_q(J_qSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type beta(betaSEXP);
    Rcpp::traits::input_parameter< const arma::vec >::type offset(offsetSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type dims(dimsSEXP);
    Rcpp::traits::input_parameter< const char* >::type family(familySEXP);
    Rcpp::traits::input_parameter< int >::type link(linkSEXP);
    Rcpp::traits::input_parameter< double >::type sig_g(sig_gSEXP);
    Rcpp::traits::input_parameter< double >::type phi(phiSEXP);
    rcpp_result_gen = Rcpp::wrap(Qfun(y, X, Z, pBigMat, group, J_q, beta, offset, dims, family, link, sig_g, phi));
    return rcpp_result_gen;
END_RCPP
}
// sig_gaus
double sig_gaus(const arma::vec& y, const arma::mat& X, const arma::mat& Z, SEXP pBigMat, const arma::vec& group, const arma::sp_mat& J_q, const arma::vec& beta, const arma::vec offset, arma::vec dims, int link);
RcppExport SEXP _glmmPen_sig_gaus(SEXP ySEXP, SEXP XSEXP, SEXP ZSEXP, SEXP pBigMatSEXP, SEXP groupSEXP, SEXP J_qSEXP, SEXP betaSEXP, SEXP offsetSEXP, SEXP dimsSEXP, SEXP linkSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::vec& >::type y(ySEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type Z(ZSEXP);
    Rcpp::traits::input_parameter< SEXP >::type pBigMat(pBigMatSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type group(groupSEXP);
    Rcpp::traits::input_parameter< const arma::sp_mat& >::type J_q(J_qSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type beta(betaSEXP);
    Rcpp::traits::input_parameter< const arma::vec >::type offset(offsetSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type dims(dimsSEXP);
    Rcpp::traits::input_parameter< int >::type link(linkSEXP);
    rcpp_result_gen = Rcpp::wrap(sig_gaus(y, X, Z, pBigMat, group, J_q, beta, offset, dims, link));
    return rcpp_result_gen;
END_RCPP
}
// phi_ml
double phi_ml(arma::vec y, arma::mat eta, int link, int limit, double eps, double phi);
RcppExport SEXP _glmmPen_phi_ml(SEXP ySEXP, SEXP etaSEXP, SEXP linkSEXP, SEXP limitSEXP, SEXP epsSEXP, SEXP phiSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type y(ySEXP);
    Rcpp::traits::input_parameter< arma::mat >::type eta(etaSEXP);
    Rcpp::traits::input_parameter< int >::type link(linkSEXP);
    Rcpp::traits::input_parameter< int >::type limit(limitSEXP);
    Rcpp::traits::input_parameter< double >::type eps(epsSEXP);
    Rcpp::traits::input_parameter< double >::type phi(phiSEXP);
    rcpp_result_gen = Rcpp::wrap(phi_ml(y, eta, link, limit, eps, phi));
    return rcpp_result_gen;
END_RCPP
}
// phi_ml_init
double phi_ml_init(arma::vec y, arma::vec eta, int link, int limit, double eps);
RcppExport SEXP _glmmPen_phi_ml_init(SEXP ySEXP, SEXP etaSEXP, SEXP linkSEXP, SEXP limitSEXP, SEXP epsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type y(ySEXP);
    Rcpp::traits::input_parameter< arma::vec >::type eta(etaSEXP);
    Rcpp::traits::input_parameter< int >::type link(linkSEXP);
    Rcpp::traits::input_parameter< int >::type limit(limitSEXP);
    Rcpp::traits::input_parameter< double >::type eps(epsSEXP);
    rcpp_result_gen = Rcpp::wrap(phi_ml_init(y, eta, link, limit, eps));
    return rcpp_result_gen;
END_RCPP
}

RcppExport SEXP _rcpp_module_boot_stan_fit4binomial_logit_model_mod();
RcppExport SEXP _rcpp_module_boot_stan_fit4gaussian_identity_model_mod();
RcppExport SEXP _rcpp_module_boot_stan_fit4poisson_log_model_mod();

static const R_CallMethodDef CallEntries[] = {
    {"_glmmPen_grp_CD_XZ_fast", (DL_FUNC) &_glmmPen_grp_CD_XZ_fast, 18},
    {"_glmmPen_soft_thresh", (DL_FUNC) &_glmmPen_soft_thresh, 2},
    {"_glmmPen_MCP_soln", (DL_FUNC) &_glmmPen_MCP_soln, 5},
    {"_glmmPen_SCAD_soln", (DL_FUNC) &_glmmPen_SCAD_soln, 5},
    {"_glmmPen_pglm_fit", (DL_FUNC) &_glmmPen_pglm_fit, 11},
    {"_glmmPen_sample_mc_inner_gibbs", (DL_FUNC) &_glmmPen_sample_mc_inner_gibbs, 10},
    {"_glmmPen_sample_mc_gibbs_adapt_rw", (DL_FUNC) &_glmmPen_sample_mc_gibbs_adapt_rw, 14},
    {"_glmmPen_invlink", (DL_FUNC) &_glmmPen_invlink, 2},
    {"_glmmPen_Qfun", (DL_FUNC) &_glmmPen_Qfun, 13},
    {"_glmmPen_sig_gaus", (DL_FUNC) &_glmmPen_sig_gaus, 10},
    {"_glmmPen_phi_ml", (DL_FUNC) &_glmmPen_phi_ml, 6},
    {"_glmmPen_phi_ml_init", (DL_FUNC) &_glmmPen_phi_ml_init, 5},
    {"_rcpp_module_boot_stan_fit4binomial_logit_model_mod", (DL_FUNC) &_rcpp_module_boot_stan_fit4binomial_logit_model_mod, 0},
    {"_rcpp_module_boot_stan_fit4gaussian_identity_model_mod", (DL_FUNC) &_rcpp_module_boot_stan_fit4gaussian_identity_model_mod, 0},
    {"_rcpp_module_boot_stan_fit4poisson_log_model_mod", (DL_FUNC) &_rcpp_module_boot_stan_fit4poisson_log_model_mod, 0},
    {NULL, NULL, 0}
};

RcppExport void R_init_glmmPen(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
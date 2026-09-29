"""Human-readable metadata and localisation for ESI descriptors/models.

Technical descriptor keys remain stable in workbooks and code.  This module
adds bilingual names, definitions, units, generation requirements and model
compatibility notes for human review.
"""
from __future__ import annotations

import re
from typing import Dict

LANGUAGE_OPTIONS = (("English", "en"),)
LANGUAGE_LABELS = [x[0] for x in LANGUAGE_OPTIONS]
LANGUAGE_MAP = {x[0]: x[1] for x in LANGUAGE_OPTIONS}

# key -> (zh name, en name, zh definition, en definition, unit/type, source, requires_3d)
_META: Dict[str, tuple] = {
    "Exact_mass": ('Monoisotopic exact mass', "Monoisotopic exact mass", 'Monoisotopic exact mass calculated from the molecular formula.', "Monoisotopic exact mass calculated from the molecular formula.", "Da", "Formula", False),
    "DBE": ('Double-bond equivalents (DBE)', "Double-bond equivalents (DBE)", 'Stoichiometric proxy for total rings and pi bonds.', "Stoichiometric proxy for total rings and pi bonds.", "dimensionless", "Formula", False),
    "Apex_RT_min": ('Apex retention time', "Apex retention time", 'Apex time of the integrated XIC peak.', "Apex time of the integrated XIC peak.", "min", "LC-MS data", False),
    "Effective_gradient_time_min": ('Effective gradient time', "Effective gradient time", 'Apex RT minus the configured gradient delay.', "Apex RT minus the configured gradient delay.", "min", "LC gradient", False),
    "Mobile_phase_A_pct": ('Mobile phase A at apex', "Mobile phase A at apex", 'Linearly interpolated mobile-phase B percentage at the effective apex time.', "Linearly interpolated mobile-phase A percentage at the effective apex time.", "%", "LC gradient", False),
    "Mobile_phase_B_pct": ('Mobile phase B at apex', "Mobile phase B at apex", 'Linearly interpolated mobile-phase B percentage at the effective apex time.', "Linearly interpolated mobile-phase B percentage at the effective apex time.", "%", "LC gradient", False),
    "B_slope_pct_per_min": ('Local B-gradient slope', "Local B-gradient slope", 'Rate of change of mobile-phase B in the gradient segment containing the apex.', "Rate of change of mobile-phase B in the gradient segment containing the apex.", "%/min", "LC gradient", False),
    "Gradient_state": ('Gradient state', "Gradient state", 'Isocratic, B-increasing or B-decreasing state; one-hot encoded before modelling.', "Isocratic, B-increasing or B-decreasing state; one-hot encoded before modelling.", "categorical", "LC gradient", False),
    "Published_Viscosity_mPa_s": ('Published eluent viscosity', "Published eluent viscosity", 'Calculated from apex organic fraction using the empirical coefficients reported with the RandFor-IE workflow.', "Calculated from apex organic fraction using the empirical coefficients reported with the RandFor-IE workflow.", "mPa\u00B7s", "Published eluent model", False),
    "Published_Surface_Tension_mN_m": ('Published eluent surface tension', "Published eluent surface tension", 'Calculated from apex organic fraction using the published cubic mixture equation.', "Calculated from apex organic fraction using the published cubic mixture equation.", "mN/m", "Published eluent model", False),
    "Published_Polarity_Index": ('Published eluent polarity index', "Published eluent polarity index", 'Linear mixture of aqueous and organic polarity indices at the apex.', "Linear mixture of aqueous and organic polarity indices at the apex.", "index", "Published eluent model", False),
    "Published_Aqueous_pH": ('Aqueous-phase pH', "Aqueous-phase pH", 'User-entered aqueous pH; 2.7 for 0.1% formic acid is an editable literature-based approximation.', "User-entered aqueous pH; 2.7 for 0.1% formic acid is an editable literature-based approximation.", "pH", "User/published eluent model", False),
    "Published_NH4_Present": ('NH4 presence', "NH4 presence", 'Binary indicator for ammonia or ammonium salts in the mobile phase.', "Binary indicator for ammonia or ammonium salts in the mobile phase.", "0/1", "User/published eluent model", False),
    "Published_Formic_Acid_pct": ('Formic acid percentage', "Formic acid percentage", 'Formic-acid percentage in the aqueous phase.', "Formic-acid percentage in the aqueous phase.", "%", "User", False),
    "FormalCharge": ('Molecular formal charge', "Molecular formal charge", 'Sum of atomic formal charges in the RDKit structure.', "Sum of atomic formal charges in the RDKit structure.", "charge", "RDKit 2D", False),
    "AbsoluteFormalChargeSum": ('Absolute formal-charge sum', "Absolute formal-charge sum", 'Sum of absolute atomic formal charges; a proxy for charge separation.', "Sum of absolute atomic formal charges; a proxy for charge separation.", "charge", "RDKit 2D", False),
    "PositiveFormalChargeAtomCount": ('Positive formal-charge atom count', "Positive formal-charge atom count", 'Number of atoms with positive formal charge.', "Number of atoms with positive formal charge.", "count", "RDKit 2D", False),
    "NegativeFormalChargeAtomCount": ('Negative formal-charge atom count', "Negative formal-charge atom count", 'Number of atoms with negative formal charge.', "Number of atoms with negative formal charge.", "count", "RDKit 2D", False),
    "HBD": ('Hydrogen-bond donors', "Hydrogen-bond donors", 'Number of hydrogen-bond donor sites.', "Number of hydrogen-bond donor sites.", "count", "RDKit 2D", False),
    "HBA": ('Hydrogen-bond acceptors', "Hydrogen-bond acceptors", 'Number of hydrogen-bond acceptor sites.', "Number of hydrogen-bond acceptor sites.", "count", "RDKit 2D", False),
    "NHOHCount": ('NH/OH group count', "NH/OH group count", 'RDKit Lipinski count of NH and OH groups.', "RDKit Lipinski count of NH and OH groups.", "count", "RDKit 2D", False),
    "NOCount": ('Nitrogen/oxygen atom count', "Nitrogen/oxygen atom count", 'Total number of nitrogen and oxygen atoms.', "Total number of nitrogen and oxygen atoms.", "count", "RDKit 2D", False),
    "AcidicSiteCount_proxy": ('Acidic-site count proxy', "Acidic-site count proxy", 'SMARTS count of selected acidic motifs; not a pKa prediction.', "SMARTS count of selected acidic motifs; not a pKa prediction.", "count/proxy", "SMARTS", False),
    "BasicSiteCount_proxy": ('Basic-site count proxy', "Basic-site count proxy", 'SMARTS count of selected basic motifs; not a pKa prediction.', "SMARTS count of selected basic motifs; not a pKa prediction.", "count/proxy", "SMARTS", False),
    "IonizableSiteCount_proxy": ('Ionizable-site count proxy', "Ionizable-site count proxy", 'Sum of acidic- and basic-site proxies.', "Sum of acidic- and basic-site proxies.", "count/proxy", "SMARTS", False),
    "BasicMinusAcidic_proxy": ('Basic-minus-acidic proxy', "Basic-minus-acidic proxy", 'Basic-site proxy minus acidic-site proxy.', "Basic-site proxy minus acidic-site proxy.", "count/proxy", "SMARTS", False),
    "ZwitterionPotential_proxy": ('Zwitterion-potential proxy', "Zwitterion-potential proxy", 'Structural indicator that both acidic and basic motifs are present.', "Structural indicator that both acidic and basic motifs are present.", "binary/proxy", "SMARTS", False),
    "CarboxylicAcidCount": ('Carboxylic-acid count', "Carboxylic-acid count", 'Count of carboxylic-acid motifs.', "Count of carboxylic-acid motifs.", "count", "SMARTS", False),
    "SulfonicAcidCount": ('Sulfonic-acid count', "Sulfonic-acid count", 'Count of sulfonic-acid motifs.', "Count of sulfonic-acid motifs.", "count", "SMARTS", False),
    "PhosphoricOHCount": ('Phosphoric OH count', "Phosphoric OH count", 'Count of phosphoric-acid OH motifs.', "Count of phosphoric-acid OH motifs.", "count", "SMARTS", False),
    "PhenolCount": ('Phenol count', "Phenol count", 'Count of phenolic OH motifs.', "Count of phenolic OH motifs.", "count", "SMARTS", False),
    "ThiolCount": ('Thiol count', "Thiol count", 'Count of thiol motifs.', "Count of thiol motifs.", "count", "SMARTS", False),
    "ImideNHCount": ('Imide NH count', "Imide NH count", 'Count of imide-NH motifs.', "Count of imide-NH motifs.", "count", "SMARTS", False),
    "AliphaticAmineCount": ('Aliphatic-amine count', "Aliphatic-amine count", 'Count of aliphatic-amine motifs.', "Count of aliphatic-amine motifs.", "count", "SMARTS", False),
    "AromaticBasicNCount": ('Aromatic basic-N count', "Aromatic basic-N count", 'Count of pyridine-like and related aromatic basic-N motifs.', "Count of pyridine-like and related aromatic basic-N motifs.", "count", "SMARTS", False),
    "AmidineGuanidineCount": ('Amidine/guanidine count', "Amidine/guanidine count", 'Count of amidine and guanidine motifs.', "Count of amidine and guanidine motifs.", "count", "SMARTS", False),
    "ImineNCount": ('Imine-N count', "Imine-N count", 'Count of imine-N motifs.', "Count of imine-N motifs.", "count", "SMARTS", False),
    "QuaternaryAmmoniumCount": ('Quaternary-ammonium count', "Quaternary-ammonium count", 'Count of quaternary-ammonium motifs.', "Count of quaternary-ammonium motifs.", "count", "SMARTS", False),
    "GasteigerChargeMax": ('Maximum Gasteiger partial charge', "Maximum Gasteiger partial charge", 'Maximum approximate atomic Gasteiger partial charge.', "Maximum approximate atomic Gasteiger partial charge.", "e (proxy)", "RDKit Gasteiger", False),
    "GasteigerChargeMin": ('Minimum Gasteiger partial charge', "Minimum Gasteiger partial charge", 'Minimum approximate atomic Gasteiger partial charge.', "Minimum approximate atomic Gasteiger partial charge.", "e (proxy)", "RDKit Gasteiger", False),
    "GasteigerChargeRange": ('Gasteiger charge range', "Gasteiger charge range", 'Difference between maximum and minimum Gasteiger partial charge.', "Difference between maximum and minimum Gasteiger partial charge.", "e (proxy)", "RDKit Gasteiger", False),
    "GasteigerAbsChargeMean": ('Mean absolute Gasteiger charge', "Mean absolute Gasteiger charge", 'Mean absolute approximate atomic Gasteiger charge.', "Mean absolute approximate atomic Gasteiger charge.", "e (proxy)", "RDKit Gasteiger", False),
    "GasteigerPositiveChargeSum": ('Positive Gasteiger charge sum', "Positive Gasteiger charge sum", 'Sum of positive approximate partial charges.', "Sum of positive approximate partial charges.", "e (proxy)", "RDKit Gasteiger", False),
    "GasteigerNegativeChargeAbsSum": ('Absolute negative Gasteiger charge sum', "Absolute negative Gasteiger charge sum", 'Absolute sum of negative approximate partial charges.', "Absolute sum of negative approximate partial charges.", "e (proxy)", "RDKit Gasteiger", False),
    "GasteigerChargeSeparation": ('Gasteiger charge separation', "Gasteiger charge separation", 'Combined proxy for positive/negative partial-charge separation.', "Combined proxy for positive/negative partial-charge separation.", "e (proxy)", "RDKit Gasteiger", False),
    "MolLogP": ('Calculated logP', "Calculated logP", 'RDKit Crippen cLogP estimate of neutral-molecule hydrophobicity.', "RDKit Crippen cLogP estimate of neutral-molecule hydrophobicity.", "dimensionless", "RDKit 2D", False),
    "MolLogP_per_HeavyAtom": ('cLogP per heavy atom', "cLogP per heavy atom", 'cLogP divided by heavy-atom count.', "cLogP divided by heavy-atom count.", "dimensionless", "Derived", False),
    "TPSA": ('Topological polar surface area', "Topological polar surface area", 'Topological polar surface area from fragment contributions.', "Topological polar surface area from fragment contributions.", "\u00C5\u00B2", "RDKit 2D", False),
    "TPSA_per_HeavyAtom": ('TPSA per heavy atom', "TPSA per heavy atom", 'TPSA divided by heavy-atom count.', "TPSA divided by heavy-atom count.", "\u00C5\u00B2/atom", "Derived", False),
    "PolarSurfaceFraction_proxy": ('Polar-surface fraction proxy', "Polar-surface fraction proxy", 'TPSA divided by Labute approximate surface area.', "TPSA divided by Labute approximate surface area.", "ratio/proxy", "Derived", False),
    "MolMR": ('Molar refractivity', "Molar refractivity", 'Crippen molar refractivity related to volume and polarizability.', "Crippen molar refractivity related to volume and polarizability.", "cm\u00B3/mol proxy", "RDKit 2D", False),
    "LabuteASA": ('Labute approximate surface area', "Labute approximate surface area", 'RDKit approximate molecular surface area.', "RDKit approximate molecular surface area.", "\u00C5\u00B2 proxy", "RDKit 2D", False),
    "FractionCSP3": ('Fraction Csp3', "Fraction Csp3", 'Fraction of carbon atoms that are sp3 hybridized.', "Fraction of carbon atoms that are sp3 hybridized.", "ratio", "RDKit 2D", False),
    "AromaticAtomFraction": ('Aromatic-atom fraction', "Aromatic-atom fraction", 'Fraction of heavy atoms that are aromatic.', "Fraction of heavy atoms that are aromatic.", "ratio", "Derived", False),
    "HeteroAtomFraction": ('Heteroatom fraction', "Heteroatom fraction", 'Fraction of heavy atoms that are heteroatoms.', "Fraction of heavy atoms that are heteroatoms.", "ratio", "Derived", False),
    "CarbonAtomFraction": ('Carbon-atom fraction', "Carbon-atom fraction", 'Fraction of heavy atoms that are carbon.', "Fraction of heavy atoms that are carbon.", "ratio", "Derived", False),
    "AromaticRingCount": ('Aromatic-ring count', "Aromatic-ring count", 'Number of aromatic rings.', "Number of aromatic rings.", "count", "RDKit 2D", False),
    "AliphaticRingCount": ('Aliphatic-ring count', "Aliphatic-ring count", 'Number of aliphatic rings.', "Number of aliphatic rings.", "count", "RDKit 2D", False),
    "RingCount": ('Total ring count', "Total ring count", 'Total number of rings recognized by RDKit.', "Total number of rings recognized by RDKit.", "count", "RDKit 2D", False),
    "RotatableBonds": ('Rotatable-bond count', "Rotatable-bond count", 'Number of Lipinski rotatable bonds.', "Number of Lipinski rotatable bonds.", "count", "RDKit 2D", False),
    "MolWt": ('Average molecular weight', "Average molecular weight", 'Molecular weight using average atomic masses.', "Molecular weight using average atomic masses.", "Da", "RDKit 2D", False),
    "HeavyAtomMolWt": ('Heavy-atom molecular weight', "Heavy-atom molecular weight", 'Molecular-weight contribution excluding hydrogen.', "Molecular-weight contribution excluding hydrogen.", "Da", "RDKit 2D", False),
    "ExactMolWt": ('RDKit exact molecular weight', "RDKit exact molecular weight", 'Monoisotopic exact molecular weight calculated from structure.', "Monoisotopic exact molecular weight calculated from structure.", "Da", "RDKit 2D", False),
    "HeavyAtomCount": ('Heavy-atom count', "Heavy-atom count", 'Number of non-hydrogen atoms.', "Number of non-hydrogen atoms.", "count", "RDKit 2D", False),
    "HeteroAtomCount": ('Heteroatom count', "Heteroatom count", 'Number of heteroatoms.', "Number of heteroatoms.", "count", "RDKit 2D", False),
    "MolVolume3D": ('3D molecular volume', "3D molecular volume", 'Grid volume from one ETKDG conformer; not a solution ensemble.', "Grid volume from one ETKDG conformer; not a solution ensemble.", "\u00C5\u00B3 proxy", "RDKit 3D", True),
    "RadiusOfGyration": ('Radius of gyration', "Radius of gyration", 'Mass-distribution size of a single 3D conformer.', "Mass-distribution size of a single 3D conformer.", "\u00C5 proxy", "RDKit 3D", True),
    "Asphericity": ('Asphericity', "Asphericity", 'Departure of a single 3D conformer from spherical shape.', "Departure of a single 3D conformer from spherical shape.", "dimensionless", "RDKit 3D", True),
    "Eccentricity": ('Eccentricity', "Eccentricity", 'Shape eccentricity of a single 3D conformer.', "Shape eccentricity of a single 3D conformer.", "dimensionless", "RDKit 3D", True),
    "SpherocityIndex": ('Spherocity index', "Spherocity index", 'Sphericity of a single 3D conformer.', "Sphericity of a single 3D conformer.", "dimensionless", "RDKit 3D", True),
    "PMI1": ('Principal moment of inertia 1', "Principal moment of inertia 1", 'Smallest principal moment of inertia.', "Smallest principal moment of inertia.", "mass\u00B7\u00C5\u00B2 proxy", "RDKit 3D", True),
    "PMI2": ('Principal moment of inertia 2', "Principal moment of inertia 2", 'Middle principal moment of inertia.', "Middle principal moment of inertia.", "mass\u00B7\u00C5\u00B2 proxy", "RDKit 3D", True),
    "PMI3": ('Principal moment of inertia 3', "Principal moment of inertia 3", 'Largest principal moment of inertia.', "Largest principal moment of inertia.", "mass\u00B7\u00C5\u00B2 proxy", "RDKit 3D", True),
    "NPR1": ('Normalized PMI 1', "Normalized PMI 1", 'PMI1 divided by PMI3.', "PMI1 divided by PMI3.", "ratio", "RDKit 3D", True),
    "NPR2": ('Normalized PMI 2', "Normalized PMI 2", 'PMI2 divided by PMI3.', "PMI2 divided by PMI3.", "ratio", "RDKit 3D", True),
    "InertialShapeFactor": ('Inertial shape factor', "Inertial shape factor", '3D shape metric derived from principal moments of inertia.', "3D shape metric derived from principal moments of inertia.", "dimensionless", "RDKit 3D", True),
    "PBF": ('Plane of best fit', "Plane of best fit", 'Atomic deviation from the best-fit plane.', "Atomic deviation from the best-fit plane.", "\u00C5 proxy", "RDKit 3D", True),
    "AmideBondCount": ('Amide-bond count', "Amide-bond count", 'Number of amide bonds.', "Number of amide bonds.", "count", "RDKit 2D", False),
    "BridgeheadAtomCount": ('Bridgehead-atom count', "Bridgehead-atom count", 'Number of bridgehead atoms.', "Number of bridgehead atoms.", "count", "RDKit 2D", False),
    "SpiroAtomCount": ('Spiro-atom count', "Spiro-atom count", 'Number of spiro atoms.', "Number of spiro atoms.", "count", "RDKit 2D", False),
    "BertzCT": ('Bertz topological complexity', "Bertz topological complexity", 'Topological complexity based on connectivity and symmetry.', "Topological complexity based on connectivity and symmetry.", "dimensionless", "RDKit 2D", False),
    "BalabanJ": ('Balaban J index', "Balaban J index", 'Topological distance index of the molecular graph.', "Topological distance index of the molecular graph.", "dimensionless", "RDKit 2D", False),
    "NumValenceElectrons": ('Valence-electron count', "Valence-electron count", 'Total number of valence electrons.', "Total number of valence electrons.", "count", "RDKit 2D", False),
    "NumRadicalElectrons": ('Radical-electron count', "Radical-electron count", 'Total number of radical electrons.', "Total number of radical electrons.", "count", "RDKit 2D", False),
    # Physically motivated engineered proxies.
    "HydrophobicGradientExposure_proxy": ('Hydrophobic-gradient exposure proxy', "Hydrophobic-gradient exposure proxy", 'cLogP multiplied by mobile-phase B fraction at the apex.', "cLogP multiplied by mobile-phase B fraction at the apex.", "proxy", "Engineered", False),
    "PolarAqueousExposure_proxy": ('Polar-aqueous exposure proxy', "Polar-aqueous exposure proxy", 'TPSA multiplied by mobile-phase A fraction at the apex.', "TPSA multiplied by mobile-phase A fraction at the apex.", "\u00C5\u00B2 proxy", "Engineered", False),
    "IonizableAqueousExposure_proxy": ('Ionizable-site aqueous-exposure proxy', "Ionizable-site aqueous-exposure proxy", 'Ionizable-site proxy multiplied by mobile-phase A fraction.', "Ionizable-site proxy multiplied by mobile-phase A fraction.", "proxy", "Engineered", False),
    "BasicAqueousExposure_proxy": ('Basic-site aqueous-exposure proxy', "Basic-site aqueous-exposure proxy", 'Basic-site proxy multiplied by mobile-phase A fraction.', "Basic-site proxy multiplied by mobile-phase A fraction.", "proxy", "Engineered", False),
    "AcidicAqueousExposure_proxy": ('Acidic-site aqueous-exposure proxy', "Acidic-site aqueous-exposure proxy", 'Acidic-site proxy multiplied by mobile-phase A fraction.', "Acidic-site proxy multiplied by mobile-phase A fraction.", "proxy", "Engineered", False),
    "ChargeDensitySurface_proxy": ('Surface charge-density proxy', "Surface charge-density proxy", 'Gasteiger charge separation divided by Labute surface area.', "Gasteiger charge separation divided by Labute surface area.", "e/\u00C5\u00B2 proxy", "Engineered", False),
    "PolarMassDensity_proxy": ('Mass-normalized polarity proxy', "Mass-normalized polarity proxy", 'TPSA divided by molecular weight.', "TPSA divided by molecular weight.", "\u00C5\u00B2/Da", "Engineered", False),
    "HydrophobicSurface_proxy": ('Hydrophobic-surface proxy', "Hydrophobic-surface proxy", 'cLogP multiplied by Labute surface area.', "cLogP multiplied by Labute surface area.", "proxy", "Engineered", False),
    "AromaticHydrophobicity_proxy": ('Aromatic-hydrophobicity proxy', "Aromatic-hydrophobicity proxy", 'cLogP multiplied by aromatic-atom fraction.', "cLogP multiplied by aromatic-atom fraction.", "proxy", "Engineered", False),
    "IonizableSiteDensity_proxy": ('Ionizable-site density proxy', "Ionizable-site density proxy", 'Ionizable-site proxy divided by heavy-atom count.', "Ionizable-site proxy divided by heavy-atom count.", "sites/atom proxy", "Engineered", False),
    "DBE_per_C": ('DBE per carbon', "DBE per carbon", 'DBE divided by carbon count.', "DBE divided by carbon count.", "ratio", "Engineered", False),
    "Hetero_to_C_ratio": ('Heteroatom-to-carbon ratio', "Heteroatom-to-carbon ratio", 'Total N/O/P/S/halogen count divided by carbon count.', "Total N/O/P/S/halogen count divided by carbon count.", "ratio", "Engineered", False),
    "Mass_per_IonizableSite_proxy": ('Mass per ionizable site proxy', "Mass per ionizable site proxy", 'Exact mass divided by at least one ionizable-site proxy.', "Exact mass divided by at least one ionizable-site proxy.", "Da/site proxy", "Engineered", False),
    "GradientChangeExposure_proxy": ('Gradient-change exposure proxy', "Gradient-change exposure proxy", 'Apex RT multiplied by local B-gradient slope.', "Apex RT multiplied by local B-gradient slope.", "% proxy", "Engineered", False),
    "LogP_TPSA_balance_proxy": ('Hydrophobic-polar balance proxy', "Hydrophobic-polar balance proxy", 'Scaled ratio of cLogP to TPSA.', "Scaled ratio of cLogP to TPSA.", "proxy", "Engineered", False),
}

_META.update({
    "Published_Organic_Formic_Acid_pct": ('Organic-phase formic acid', 'Organic-phase formic acid', 'Organic-phase formic-acid percentage.', 'Organic-phase formic-acid percentage.', "%", "User", False),
    "Published_Apex_Formic_Acid_pct": ('Apex formic-acid fraction', 'Apex formic-acid fraction', 'Programmed formic-acid percentage at the chromatographic apex.', 'Programmed formic-acid percentage at the chromatographic apex.', "%", "LC gradient", False),
    "NegESI_DeprotonatableSiteCount_proxy": ('Deprotonatable-site proxy', 'Deprotonatable-site proxy', 'Count of transparent acidic structural motifs.', 'Count of transparent acidic structural motifs.', "count/proxy", "SMARTS", False),
    "NegESI_StrongAcidSiteCount_proxy": ('Strong-acid-site proxy', 'Strong-acid-site proxy', 'Sulfonic and phosphoric acidic-site count proxy.', 'Sulfonic and phosphoric acidic-site count proxy.', "count/proxy", "SMARTS", False),
    "NegESI_MediumAcidSiteCount_proxy": ('Medium-acid-site proxy', 'Medium-acid-site proxy', 'Carboxylic-acid and imide-NH count proxy.', 'Carboxylic-acid and imide-NH count proxy.', "count/proxy", "SMARTS", False),
    "NegESI_WeakAcidSiteCount_proxy": ('Weak-acid-site proxy', 'Weak-acid-site proxy', 'Phenol and thiol count proxy.', 'Phenol and thiol count proxy.', "count/proxy", "SMARTS", False),
    "NegESI_pKaClass_proxy": ('Acid-class pKa proxy', 'Acid-class pKa proxy', 'Coarse acid-class value; not a predicted pKa.', 'Coarse acid-class value; not a predicted pKa.', "proxy", "Rule based", False),
    "NegESI_pH_minus_pKa_proxy": ('pH-minus-acid-class proxy', 'pH-minus-acid-class proxy', 'Aqueous pH minus the coarse acid-class proxy.', 'Aqueous pH minus the coarse acid-class proxy.', "proxy", "Rule based", False),
    "NegESI_SolutionIonizedFraction_proxy": ('Solution ionized-fraction proxy', 'Solution ionized-fraction proxy', 'Henderson-Hasselbalch-like proxy based on the coarse acid class.', 'Henderson-Hasselbalch-like proxy based on the coarse acid class.', "0-1 proxy", "Rule based", False),
    "NegESI_EWGCount_proxy": ('Electron-withdrawing-group proxy', 'Electron-withdrawing-group proxy', 'Count or elemental fallback proxy for electron-withdrawing motifs.', 'Count or elemental fallback proxy for electron-withdrawing motifs.', "count/proxy", "SMARTS/formula", False),
    "NegESI_AcidicSiteDensity_proxy": ('Acidic-site density proxy', 'Acidic-site density proxy', 'Deprotonatable-site proxy divided by heavy-atom count.', 'Deprotonatable-site proxy divided by heavy-atom count.', "proxy", "Engineered", False),
    "NegESI_AnionChargeDelocalization_proxy": ('Anion charge-delocalization proxy', 'Anion charge-delocalization proxy', 'Gasteiger charge, aromaticity and EWG based proxy; not a quantum-chemical descriptor.', 'Gasteiger charge, aromaticity and EWG based proxy; not a quantum-chemical descriptor.', "proxy", "Engineered", False),
    "NegESI_AnionStability_proxy": ('Anion stability proxy', 'Anion stability proxy', 'Ionized-fraction, delocalization and hydrophobicity interaction proxy.', 'Ionized-fraction, delocalization and hydrophobicity interaction proxy.', "proxy", "Engineered", False),
    "NegESI_IonizationDelocalization_proxy": ('Ionization-delocalization interaction', 'Ionization-delocalization interaction', 'Product of ionized-fraction and charge-delocalization proxies.', 'Product of ionized-fraction and charge-delocalization proxies.', "proxy", "Engineered", False),
    "NegESI_OrganicFractionDelocalization_proxy": ('Organic-fraction delocalization', 'Organic-fraction delocalization', 'Apex organic fraction multiplied by the charge-delocalization proxy.', 'Apex organic fraction multiplied by the charge-delocalization proxy.', "proxy", "Engineered", False),
    "NegESI_SurfaceTensionChargeRelease_proxy": ('Surface-tension charge-release proxy', 'Surface-tension charge-release proxy', 'Charge-delocalization term normalized by apex surface tension.', 'Charge-delocalization term normalized by apex surface tension.', "proxy", "Engineered", False),
    "NegESI_ViscosityDesolvation_proxy": ('Viscosity-desolvation proxy', 'Viscosity-desolvation proxy', 'Hydrophobicity term normalized by apex mobile-phase viscosity.', 'Hydrophobicity term normalized by apex mobile-phase viscosity.', "proxy", "Engineered", False),
    "NegESI_PolarityIonization_proxy": ('Polarity-ionization proxy', 'Polarity-ionization proxy', 'Apex polarity index multiplied by the solution-ionized-fraction proxy.', 'Apex polarity index multiplied by the solution-ionized-fraction proxy.', "proxy", "Engineered", False),
    "NegESI_HydrophobicIonRelease_proxy": ('Hydrophobic ion-release proxy', 'Hydrophobic ion-release proxy', 'cLogP, organic fraction and charge-delocalization interaction proxy.', 'cLogP, organic fraction and charge-delocalization interaction proxy.', "proxy", "Engineered", False),
    "ElectronWithdrawingGroupCount_proxy": ('Electron-withdrawing-group count proxy', 'Electron-withdrawing-group count proxy', 'SMARTS sum of selected carbonyl, nitro, nitrile, oxidized sulfur and halogen motifs.', 'SMARTS sum of selected carbonyl, nitro, nitrile, oxidized sulfur and halogen motifs.', "count/proxy", "SMARTS", False),
})

# Experimental ion-channel behavior derived from multi-channel XIC evidence.
_META.update({
    "Observed_Fragility_Index": ('Observed fragility index', "Observed fragility index", 'Fraction of accepted in-source fragment/diagnostic area in the total accepted primary, adduct/cluster and fragment evidence. It is source-condition dependent, not an intrinsic bond energy.', "Fraction of accepted in-source fragment/diagnostic area in the total accepted primary, adduct/cluster and fragment evidence. It is source-condition dependent, not an intrinsic bond energy.", "0-1", "Multi-channel XIC evidence", False),
    "Adduct_Cluster_Proneness_Index": ('Adduct/cluster proneness index', "Adduct/cluster proneness index", 'Fraction of accepted adduct, cluster and substitution-channel area among all accepted ion-channel evidence.', "Fraction of accepted adduct, cluster and substitution-channel area among all accepted ion-channel evidence.", "0-1", "Multi-channel XIC evidence", False),
    "Primary_Ion_Fraction": ('Primary-ion fraction', "Primary-ion fraction", '[M-H]- area divided by all accepted ion-channel evidence area.', "[M-H]- area divided by all accepted ion-channel evidence area.", "0-1", "Multi-channel XIC evidence", False),
    "Ion_Form_Diversity_Count": ('Ion-form diversity count', "Ion-form diversity count", 'Number of accepted primary, adduct, cluster and diagnostic ion channels for a compound.', "Number of accepted primary, adduct, cluster and diagnostic ion channels for a compound.", "count", "Multi-channel XIC evidence", False),
    "Accepted_Channel_Count": ('Accepted channel count', "Accepted channel count", 'Number of additional negative-ion channels passing height, coelution, shape and isotope-support rules.', "Number of additional negative-ion channels passing height, coelution, shape and isotope-support rules.", "count", "Multi-channel XIC evidence", False),
    "Summed_Channel_Count": ('Summed channel count', "Summed channel count", 'Number of accepted panel channels marked sum and added to quantitative area.', "Number of accepted panel channels marked sum and added to quantitative area.", "count", "Multi-channel XIC evidence", False),
    "Additional_Summed_Area": ('Additional summed area', "Additional summed area", 'Additional peak area contributed by fixed-panel channels beyond the main [M-H]- channel.', "Additional peak area contributed by fixed-panel channels beyond the main [M-H]- channel.", "area", "Multi-channel XIC evidence", False),
    "Br_Fragment_Fraction": ('Bromine-fragment fraction', "Bromine-fragment fraction", 'Fraction of accepted debromination/Br- diagnostic evidence among total ion-channel evidence for brominated formulas.', "Fraction of accepted debromination/Br- diagnostic evidence among total ion-channel evidence for brominated formulas.", "0-1", "Multi-channel XIC evidence", False),
})

_GROUP_ZH = {
    "Component identity": 'Component identity',
    "LC gradient / RT": 'LC gradient / RT',
    "Formula / exact mass": 'Formula / exact mass',
    "3D size / shape": '3D size / shape',
    "Acid/base / charge": 'Acid/base / charge',
    "Extended RDKit": 'Extended RDKit',
    "Hydrophobicity / polarity": 'Hydrophobicity / polarity',
    "Size / topology": 'Size / topology',
    "Engineered physical proxy": 'Engineered physical proxy',
    "Published IE eluent descriptors": 'Published IE eluent descriptors',
    "Experimental ion behavior": 'Experimental ion behavior',
    "Experimental_ion_behavior": 'Experimental ion behavior',
}

_STATUS_ZH = {
    "Selected": 'Selected',
    "Candidate": 'Candidate',
    "Not selected": 'Not selected',
    "Redundant": 'Redundant',
    "Constant": 'Constant',
    "Unavailable": 'Unavailable',
    "Disabled": 'Disabled',
    "Generation failed": 'Generation failed',
    "Too sparse": 'Too sparse',
}


def descriptor_meta(name: str) -> Dict[str, object]:
    """Return bilingual metadata for a descriptor technical key."""
    name = str(name or "")
    if name in _META:
        zh, en, dzh, den, unit, source, req3d = _META[name]
    elif name.startswith("FormulaFrac_"):
        el = name.split("_", 1)[1]
        zh, en = f'{el} atomic fraction', f"Atomic fraction of {el}"
        dzh, den = f'{el} atom count divided by total formula atom count.', f"Number of {el} atoms divided by total atom count in the formula."
        unit, source, req3d = "ratio", "Formula", False
    elif name.startswith("Formula_"):
        el = name.split("_", 1)[1]
        zh, en = f'{el} atom count', f"{el} atom count"
        dzh, den = f'Number of {el} atoms in the molecular formula; zero if absent.', f"Number of {el} atoms in the molecular formula; zero when absent."
        unit, source, req3d = "count", "Formula", False
    elif name.startswith(("PEOE_VSA", "SlogP_VSA", "SMR_VSA", "EState_VSA", "BCUT2D_")):
        zh, en = name, name
        dzh = 'Extended RDKit 2D descriptor; useful for machine learning but with limited standalone physical interpretation.'
        den = "Extended RDKit 2D descriptor; useful for machine learning but with limited standalone physical interpretation."
        unit, source, req3d = "numeric", "RDKit extended", False
    elif name in {"A_Formula", "B_Formula", "C_Formula"}:
        role = name[0]
        zh, en = f'{role} component formula category', f"Component {role} formula category"
        dzh = 'Optional categorical feature; one-hot encoded to numeric 0/1 columns before modelling.'
        den = "Optional categorical feature; one-hot encoded to numeric 0/1 columns before modelling."
        unit, source, req3d = "categorical", "Input Combo", False
    else:
        zh = en = name
        dzh = 'Numeric descriptor; consult the generating module or RDKit documentation for the exact definition.'
        den = "Numeric descriptor; consult the generating module or RDKit documentation for the exact definition."
        unit, source, req3d = "numeric", "Generated", False
    return {
        "Descriptor": name,
        "Name_zh": zh,
        "Name_en": en,
        "Definition_zh": dzh,
        "Definition_en": den,
        "Unit_or_type": unit,
        "Source": source,
        "Requires_3D": "Yes" if req3d else "No",
    }


def display_name(name: str, language: str = "en") -> str:
    meta = descriptor_meta(name)
    return str(meta["Name_zh"] if str(language).lower().startswith("zh") else meta["Name_en"])


def group_display(group: str, language: str = "en") -> str:
    if str(language).lower().startswith("zh"):
        return _GROUP_ZH.get(str(group), str(group))
    return str(group)


def status_display(status: str, language: str = "en") -> str:
    if str(language).lower().startswith("zh"):
        return _STATUS_ZH.get(str(status), str(status))
    return str(status)


def model_compatibility(feature_type: str, group: str = "") -> str:
    """Human-readable compatibility statement for classic models."""
    if str(feature_type).lower() == "categorical":
        return "One-hot -> numeric; usable by all models, but may overfit in small samples"
    return "Numeric after imputation; scaled for BayesianRidge/SVR/KNN/PLS/GP, unscaled for tree models"


def clean_label(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()

# ---------------------------------------------------------------------------
# Mechanistic review metadata used by the manual descriptor selector.
# ---------------------------------------------------------------------------

ADVANCED_EXTERNAL_DESCRIPTORS = (
    "External_pKa",
    "External_IonizedFraction",
    "External_WAPS",
    "External_COSMO_SigmaMoment1",
    "External_COSMO_SigmaMoment2",
    "External_COSMO_SurfaceArea",
    "External_SolvationFreeEnergy",
    "External_DipoleMoment",
    "External_Polarizability",
    "External_AnionStabilizationEnergy",
)


def _mechanistic_group(name: str) -> str:
    name = str(name or "")
    if name in {
        "Observed_Fragility_Index",
        "Adduct_Cluster_Proneness_Index",
        "Primary_Ion_Fraction",
        "Ion_Form_Diversity_Count",
        "Accepted_Channel_Count",
        "Summed_Channel_Count",
        "Additional_Summed_Area",
        "Br_Fragment_Fraction",
    }:
        return "Experimental ion behavior"
    if name in ADVANCED_EXTERNAL_DESCRIPTORS or name.startswith("External_"):
        return "Quantum/COSMO external"
    if name.startswith("NegESI_"):
        return "Negative-ESI mechanism"
    if name.startswith("Published_"):
        return "Mobile-phase physics"
    if name in {"Apex_RT_min", "Effective_gradient_time_min", "Mobile_phase_A_pct", "Mobile_phase_B_pct", "B_slope_pct_per_min", "Gradient_state"}:
        return "Experimental LC condition"
    if name.startswith(("Formula_", "FormulaFrac_")) or name in {"Exact_mass", "DBE"}:
        return "Coarse composition"
    if name in _THREE_D_FEATURE_NAMES:
        return "3D geometry"
    if any(token in name for token in (
        "Gasteiger", "FormalCharge", "Acid", "Basic", "Ioniz", "Amine", "Phenol", "Thiol", "Imide", "HBD", "HBA", "NHOH", "NOCount",
    )):
        return "Ionization/charge"
    if any(token in name for token in (
        "LogP", "TPSA", "Surface", "ASA", "MolMR", "Hydrophobic", "Polar", "Aromatic", "Hetero", "Ring", "Rotatable", "FractionCSP3",
    )):
        return "Physicochemical structure"
    if name.endswith("_proxy"):
        return "Engineered mechanism proxy"
    return "Topology/other"


_THREE_D_FEATURE_NAMES = {
    "MolVolume3D", "RadiusOfGyration", "Asphericity", "Eccentricity",
    "SpherocityIndex", "PMI1", "PMI2", "PMI3", "NPR1", "NPR2",
    "InertialShapeFactor", "PBF",
}


def mechanistic_profile(name: str) -> Dict[str, object]:
    """Return bilingual mechanistic interpretation for an ESI descriptor.

    The direction fields are intentionally qualitative.  Most ESI effects are
    conditional on ion mode, pH, solvent composition, source geometry and
    chemical class, so a universal positive/negative coefficient is not
    claimed.
    """
    name = str(name or "")
    group = _mechanistic_group(name)
    level = "Intermediate"
    automatic = "Yes"
    zh = 'This descriptor provides structural or experimental context; its direction must be established by out-of-fold validation.'
    en = "This descriptor provides structural or experimental context; its direction must be established by out-of-fold validation."
    direction_zh = 'Context dependent'
    direction_en = "Context dependent"

    if group == "Experimental ion behavior":
        level = "Experimental / source-dependent"
        automatic = "Yes"
        zh = 'This descriptor is derived from accepted primary, adduct/cluster, in-source fragment, or diagnostic ion channels near the same chromatographic peak. It characterizes ion-form partitioning and apparent fragility under the current LC/ESI source conditions, but it is not an intrinsic bond energy, thermodynamic dissociation energy, or instrument-independent molecular constant.'
        en = "This descriptor is derived from accepted primary, adduct/cluster, in-source fragment, or diagnostic ion channels near the same chromatographic peak. It characterizes ion-form partitioning and apparent fragility under the current LC/ESI source conditions, but it is not an intrinsic bond energy, thermodynamic dissociation energy, or instrument-independent molecular constant."
        direction_zh = 'Context dependent; requires out-of-fold validation'
        direction_en = "Context dependent; requires out-of-fold validation"
    elif group == "Quantum/COSMO external":
        level = "Mechanistic / external"
        automatic = "No"
        if name == "External_pKa":
            zh = 'Together with mobile-phase pH, pKa controls solution deprotonation and is one of the most direct mechanistic variables for negative ESI. It requires experiment, a validated predictor, or quantum-chemical software.'
            en = "Together with mobile-phase pH, pKa controls solution deprotonation and is one of the most direct mechanistic variables for negative ESI. It requires experiment, a validated predictor, or quantum-chemical software."
            direction_zh, direction_en = 'Usually favorable when pH exceeds pKa', "Usually favorable when pH exceeds pKa"
        elif name == "External_IonizedFraction":
            zh = 'Fraction ionized under the actual eluent conditions; closer to the ion-formation mechanism than simple acidic-site counts.'
            en = "Fraction ionized under the actual eluent conditions; closer to the ion-formation mechanism than simple acidic-site counts."
            direction_zh, direction_en = 'Usually positive', "Usually positive"
        elif name == "External_WAPS":
            zh = 'WAPS is a COSMO surface-charge descriptor of anion charge delocalization; published negative-ESI models combine it with solution ionization degree to explain ionization efficiency.'
            en = "WAPS is a COSMO surface-charge descriptor of anion charge delocalization; published negative-ESI models combine it with solution ionization degree to explain ionization efficiency."
            direction_zh, direction_en = 'More favorable charge delocalization is usually positive', "More favorable charge delocalization is usually positive"
        elif "COSMO" in name or "Solvation" in name or "Anion" in name:
            zh = 'This quantum/solvation descriptor can represent charge delocalization, solvation stabilization, or anion-formation energy; RDKit cannot generate it reliably.'
            en = "This quantum/solvation descriptor can represent charge delocalization, solvation stabilization, or anion-formation energy; RDKit cannot generate it reliably."
        else:
            zh = 'This advanced physical descriptor requires external quantum-chemical or molecular-simulation software and can be imported but not generated here.'
            en = "This advanced physical descriptor requires external quantum-chemical or molecular-simulation software and can be imported but not generated here."
    elif group == "Negative-ESI mechanism":
        level = "Mechanistic proxy"
        if "SolutionIonizedFraction" in name or "pH_minus_pKa" in name:
            zh = 'Approximates solution deprotonation from aqueous pH and coarse acid-class rules; more mechanistic than elemental counts but not a measured pKa.'
            en = "Approximates solution deprotonation from aqueous pH and coarse acid-class rules; more mechanistic than elemental counts but not a measured pKa."
            direction_zh, direction_en = 'Usually positive', "Usually positive"
        elif "Delocalization" in name or "AnionStability" in name:
            zh = 'Approximates anion charge delocalization/stability from partial charge, aromaticity and electron-withdrawing motifs; true WAPS or quantum calculations are more reliable.'
            en = "Approximates anion charge delocalization/stability from partial charge, aromaticity and electron-withdrawing motifs; true WAPS or quantum calculations are more reliable."
            direction_zh, direction_en = 'Usually positive', "Usually positive"
        elif "Viscosity" in name:
            zh = 'Proxy for the effect of eluent viscosity on droplet evaporation and desolvation; higher viscosity usually impedes rapid desolvation.'
            en = "Proxy for the effect of eluent viscosity on droplet evaporation and desolvation; higher viscosity usually impedes rapid desolvation."
            direction_zh, direction_en = 'Usually negative', "Usually negative"
        elif "SurfaceTension" in name:
            zh = 'Proxy for surface-tension effects on charge release and ion-evaporation barriers; direction depends on droplet composition and source conditions.'
            en = "Proxy for surface-tension effects on charge release and ion-evaporation barriers; direction depends on droplet composition and source conditions."
        elif "Hydrophobic" in name or "OrganicFraction" in name:
            zh = 'Proxy for hydrophobic surface enrichment and organic-modifier-enhanced evaporation; excessive hydrophobicity or poor solubility can reverse the effect.'
            en = "Proxy for hydrophobic surface enrichment and organic-modifier-enhanced evaporation; excessive hydrophobicity or poor solubility can reverse the effect."
        else:
            zh = 'Negative-ESI mechanistic proxy combining acidity, partial charge, hydrophobicity and eluent conditions; it requires grouped out-of-fold validation.'
            en = "Negative-ESI mechanistic proxy combining acidity, partial charge, hydrophobicity and eluent conditions; it requires grouped out-of-fold validation."
    elif group == "Mobile-phase physics":
        level = "Mechanistic condition"
        if "Viscosity" in name:
            zh = 'Apex mixed-eluent viscosity proxy affecting droplet formation, mass transfer and desolvation.'
            en = "Apex mixed-eluent viscosity proxy affecting droplet formation, mass transfer and desolvation."
            direction_zh, direction_en = 'Higher values are usually unfavorable', "Higher values are usually unfavorable"
        elif "Surface_Tension" in name:
            zh = 'Apex mixed-eluent surface-tension proxy affecting droplet fission and ion-release barriers.'
            en = "Apex mixed-eluent surface-tension proxy affecting droplet fission and ion-release barriers."
            direction_zh, direction_en = 'Usually context dependent', "Usually context dependent"
        elif "Polarity" in name:
            zh = 'Apex solvent-polarity proxy affecting solvation, surface partitioning and anion stability.'
            en = "Apex solvent-polarity proxy affecting solvation, surface partitioning and anion stability."
        elif "pH" in name:
            zh = 'Aqueous pH proxy; together with analyte pKa it controls deprotonation. The default for 0.1% formic acid is only an approximation.'
            en = "Aqueous pH proxy; together with analyte pKa it controls deprotonation. The default for 0.1% formic acid is only an approximation."
            direction_zh, direction_en = 'Usually favorable as pH rises relative to pKa', "Usually favorable as pH rises relative to pKa"
        else:
            zh = 'Published-style eluent property calculated at the chromatographic apex to include LC conditions in the ionization-efficiency model.'
            en = "Published-style eluent property calculated at the chromatographic apex to include LC conditions in the ionization-efficiency model."
    elif group == "Experimental LC condition":
        level = "Experimental proxy"
        zh = 'Retention time, apex organic fraction and gradient slope are empirical proxies for hydrophobicity and the instantaneous eluent environment; they are not intrinsic molecular properties.'
        en = "Retention time, apex organic fraction and gradient slope are empirical proxies for hydrophobicity and the instantaneous eluent environment; they are not intrinsic molecular properties."
    elif group == "Ionization/charge":
        level = "Mechanistic proxy"
        if "Gasteiger" in name:
            zh = 'Gasteiger partial charges approximate local electron distribution and can indicate charge localization/delocalization after deprotonation; they are not quantum-chemical charges.'
            en = "Gasteiger partial charges approximate local electron distribution and can indicate charge localization/delocalization after deprotonation; they are not quantum-chemical charges."
        elif "Acid" in name or "Ioniz" in name or "Phenol" in name or "Thiol" in name:
            zh = 'SMARTS-based potential deprotonation-site descriptor; distinguishes acidic motifs but cannot replace pKa.'
            en = "SMARTS-based potential deprotonation-site descriptor; distinguishes acidic motifs but cannot replace pKa."
        else:
            zh = 'Describes formal charge, hydrogen bonding or ionizable sites that may affect solution ion formation and desolvation.'
            en = "Describes formal charge, hydrogen bonding or ionizable sites that may affect solution ion formation and desolvation."
    elif group == "Physicochemical structure":
        level = "Physicochemical"
        if "LogP" in name or "Hydrophobic" in name:
            zh = 'Hydrophobicity can promote enrichment at the ESI droplet surface, but it also relates to solubility and coelution suppression; the direction is not globally fixed.'
            en = "Hydrophobicity can promote enrichment at the ESI droplet surface, but it also relates to solubility and coelution suppression; the direction is not globally fixed."
        elif "TPSA" in name or "HBA" in name or "HBD" in name:
            zh = 'Polarity and hydrogen bonding affect solvation and droplet-surface partitioning; strong hydrogen bonding may retain molecules in the droplet bulk and reduce ion release.'
            en = "Polarity and hydrogen bonding affect solvation and droplet-surface partitioning; strong hydrogen bonding may retain molecules in the droplet bulk and reduce ion release."
        elif "Surface" in name or "ASA" in name or "MolMR" in name:
            zh = 'Surface area, refractivity and size proxies affect droplet-surface occupancy, solvation and gas-phase ion formation.'
            en = "Surface area, refractivity and size proxies affect droplet-surface occupancy, solvation and gas-phase ion formation."
        else:
            zh = 'Describes polarity, aromaticity, ring systems and flexibility; useful for chemical-class context but usually interpreted jointly with acid/base and eluent variables.'
            en = "Describes polarity, aromaticity, ring systems and flexibility; useful for chemical-class context but usually interpreted jointly with acid/base and eluent variables."
    elif group == "3D geometry":
        level = "3D physicochemical"
        zh = 'Volume and shape proxy from one computed conformer; it may affect surface activity and desolvation but does not represent the full solution conformational ensemble.'
        en = "Volume and shape proxy from one computed conformer; it may affect surface activity and desolvation but does not represent the full solution conformational ensemble."
    elif group == "Coarse composition":
        level = "Coarse covariate"
        zh = 'Captures only elemental composition, molecular size or unsaturation; useful as a chemical-space covariate but insufficient to explain negative-ESI ionization mechanism by itself.'
        en = "Captures only elemental composition, molecular size or unsaturation; useful as a chemical-space covariate but insufficient to explain negative-ESI ionization mechanism by itself."
        direction_zh, direction_en = 'No universal direction', "No universal direction"
    else:
        zh = 'Topological/statistical descriptor that separates chemical structures but is usually less mechanistic than pKa, charge delocalization and eluent properties.'
        en = "Topological/statistical descriptor that separates chemical structures but is usually less mechanistic than pKa, charge delocalization and eluent properties."

    return {
        "Mechanistic_group": group,
        "Mechanistic_level": level,
        "Theory_zh": zh,
        "Theory_en": en,
        "Expected_direction_zh": direction_zh,
        "Expected_direction_en": direction_en,
        "Auto_generated": automatic,
    }

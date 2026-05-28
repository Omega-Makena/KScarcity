"""
Theory-grounded typed relationship ground truth for East African macro data.

Sources:
    - IMF Article IV Consultation reports (Kenya 2019, 2022; Tanzania 2021; Uganda 2020)
    - World Bank Development Indicators methodology notes
    - Blanchard (2017) Macroeconomics, 7th ed.
    - Mankiw (2019) Macroeconomics, 10th ed.
    - Obstfeld & Rogoff (1996) Foundations of International Macroeconomics
"""
from __future__ import annotations


def get_typed_ground_truth(
    exclude_missing_vars: set[str] | None = None,
) -> list[dict]:
    """
    27 theory-grounded typed relationships with economic citations.

    Args:
        exclude_missing_vars: if provided, any GT entry that references a variable
            in this set (source, target, mediator, or moderator) is excluded.
            Prints a report of which entries were excluded and why.
    """
    all_entries = _get_all_gt_entries()
    if not exclude_missing_vars:
        return all_entries

    included = []
    excluded = []
    for entry in all_entries:
        entry_vars = {entry['source'], entry['target']}
        entry_vars.update(entry.get(k, '') for k in ('mediator', 'moderator') if k in entry)
        missing = entry_vars & exclude_missing_vars
        if missing:
            excluded.append((entry, missing))
        else:
            included.append(entry)

    if excluded:
        print(f'  GT exclusions due to missing variables ({len(excluded)} entries):')
        for entry, missing in excluded:
            src, tgt = entry['source'], entry['target']
            rel_type = entry['type']
            print(f'    EXCLUDED [{rel_type}] {src} -> {tgt} '
                  f'(missing: {", ".join(sorted(missing))})')
    return included


def _get_all_gt_entries() -> list[dict]:
    """Raw list of all 27 GT entries (no filtering)."""
    return [
        # ==============================================================
        # TEMPORAL — autoregressive persistence
        # ==============================================================
        {
            'source': 'inflation_cpi', 'target': 'inflation_cpi',
            'type': 'temporal', 'expected_sign': +1, 'strength': 'strong',
            'theoretical_basis': 'Inflation persistence — adaptive expectations, wage-price spirals '
                                 '(Blanchard Ch.8; Gordon 1990)',
        },
        {
            'source': 'gdp_growth', 'target': 'gdp_growth',
            'type': 'temporal', 'expected_sign': +1, 'strength': 'strong',
            'theoretical_basis': 'GDP growth mean-reversion — business cycle persistence '
                                 '(Hamilton 1989; Blanchard Ch.9)',
        },
        {
            'source': 'unemployment', 'target': 'unemployment',
            'type': 'temporal', 'expected_sign': +1, 'strength': 'strong',
            'theoretical_basis': 'Unemployment hysteresis — labor market frictions '
                                 '(Blanchard & Summers 1986)',
        },
        {
            'source': 'real_interest_rate', 'target': 'real_interest_rate',
            'type': 'temporal', 'expected_sign': +1, 'strength': 'strong',
            'theoretical_basis': 'Interest rate smoothing — central bank gradualism '
                                 '(Woodford 2003; Taylor 1993)',
        },

        # ==============================================================
        # CAUSAL — directed lagged effects
        # ==============================================================
        {
            'source': 'inflation_cpi', 'target': 'real_interest_rate',
            'type': 'causal', 'expected_sign': +1, 'strength': 'strong',
            'theoretical_basis': 'Taylor Rule — CB raises nominal rate more than 1:1 with '
                                 'inflation (Taylor 1993; Clarida et al. 1999)',
        },
        {
            'source': 'gdp_growth', 'target': 'unemployment',
            'type': 'causal', 'expected_sign': -1, 'strength': 'strong',
            'theoretical_basis': "Okun's Law — output gap reduces unemployment "
                                 '(Okun 1962; Ball et al. 2017)',
        },
        {
            'source': 'real_interest_rate', 'target': 'private_credit',
            'type': 'causal', 'expected_sign': -1, 'strength': 'strong',
            'theoretical_basis': 'Credit channel — higher rates reduce lending '
                                 '(Bernanke & Gertler 1995; Mishkin 1996)',
        },
        {
            'source': 'govt_debt', 'target': 'real_interest_rate',
            'type': 'causal', 'expected_sign': +1, 'strength': 'moderate',
            'theoretical_basis': 'Crowding-out — fiscal deficits raise rates '
                                 '(Mankiw Ch.3; Laubach 2009)',
        },
        {
            'source': 'private_credit', 'target': 'gdp_growth',
            'type': 'causal', 'expected_sign': +1, 'strength': 'moderate',
            'theoretical_basis': 'Financial deepening — credit enables investment '
                                 '(King & Levine 1993; Rajan & Zingales 1998)',
        },
        {
            'source': 'govt_consumption', 'target': 'gdp_growth',
            'type': 'causal', 'expected_sign': +1, 'strength': 'moderate',
            'theoretical_basis': 'Fiscal multiplier — government spending raises output '
                                 '(Blanchard & Perotti 2002; Ramey 2011)',
        },

        # ==============================================================
        # CORRELATIONAL — bidirectional, common-factor co-movement
        # ==============================================================
        {
            'source': 'exports_gdp', 'target': 'imports_gdp',
            'type': 'correlational', 'expected_sign': +1, 'strength': 'strong',
            'theoretical_basis': 'Trade openness — both driven by liberalisation, FX regime, '
                                 'global demand (Obstfeld & Rogoff Ch.1; Frankel & Romer 1999)',
        },
        {
            'source': 'gdp_growth', 'target': 'gcf',
            'type': 'correlational', 'expected_sign': +1, 'strength': 'strong',
            'theoretical_basis': 'Accelerator — investment and output co-move; direction is '
                                 'simultaneous (Samuelson 1939; Clark 1917)',
        },
        {
            'source': 'electricity_access', 'target': 'internet_users',
            'type': 'correlational', 'expected_sign': +1, 'strength': 'strong',
            'theoretical_basis': 'Infrastructure co-development — electricity is prerequisite '
                                 'for internet; both driven by development level '
                                 '(World Bank WDI methodology; ITU 2020)',
        },
        {
            'source': 'school_enrollment', 'target': 'life_expectancy',
            'type': 'correlational', 'expected_sign': +1, 'strength': 'moderate',
            'theoretical_basis': 'Human capital co-movement — education and health driven by '
                                 'development spending (Sen 1999; UNDP HDI methodology)',
        },

        # ==============================================================
        # COMPETITIVE — negative-sum trade-offs
        # ==============================================================
        {
            'source': 'govt_consumption', 'target': 'private_credit',
            'type': 'competitive', 'expected_sign': -1, 'strength': 'moderate',
            'theoretical_basis': 'Fiscal crowding-out of private sector — government borrowing '
                                 'absorbs domestic credit (Mankiw Ch.3; Friedman 1978)',
        },
        {
            'source': 'imports_gdp', 'target': 'current_account',
            'type': 'competitive', 'expected_sign': -1, 'strength': 'strong',
            'theoretical_basis': 'Balance of payments identity — imports reduce CA balance '
                                 'by construction (Obstfeld & Rogoff Ch.1)',
        },

        # ==============================================================
        # COMPOSITIONAL — parts-of-whole accounting identities
        # ==============================================================
        {
            'source': 'exports_gdp', 'target': 'current_account',
            'type': 'compositional', 'expected_sign': +1, 'strength': 'strong',
            'theoretical_basis': 'CA = Exports - Imports + Net income + Net transfers '
                                 '(BPM6; Obstfeld & Rogoff Ch.1)',
        },
        {
            'source': 'govt_consumption', 'target': 'gdp_growth',
            'type': 'compositional', 'expected_sign': +1, 'strength': 'strong',
            'theoretical_basis': 'GDP = C + I + G + NX; govt consumption is a GDP component '
                                 '(SNA 2008; Mankiw Ch.2)',
        },
        {
            'source': 'gcf', 'target': 'gdp_growth',
            'type': 'compositional', 'expected_sign': +1, 'strength': 'strong',
            'theoretical_basis': 'GDP = C + I + G + NX; GCF is investment component '
                                 '(SNA 2008; Mankiw Ch.2)',
        },

        # ==============================================================
        # EQUILIBRIUM — mean-reverting / error-correction pairs
        # ==============================================================
        {
            'source': 'current_account', 'target': 'current_account',
            'type': 'equilibrium', 'expected_sign': 0, 'strength': 'moderate',
            'theoretical_basis': 'Intertemporal CA sustainability — CA/GDP reverts to '
                                 'sustainable deficit level (Obstfeld & Rogoff Ch.2; '
                                 'Trehan & Walsh 1991)',
        },
        {
            'source': 'real_interest_rate', 'target': 'real_interest_rate',
            'type': 'equilibrium', 'expected_sign': 0, 'strength': 'moderate',
            'theoretical_basis': 'Natural rate of interest — real rate reverts to r* '
                                 '(Wicksell 1898; Laubach & Williams 2003)',
        },

        # ==============================================================
        # MEDIATING — indirect paths X -> M -> Y
        # ==============================================================
        {
            'source': 'inflation_cpi', 'mediator': 'real_interest_rate',
            'target': 'private_credit',
            'type': 'mediating', 'expected_sign': -1, 'strength': 'strong',
            'theoretical_basis': 'Monetary transmission — inflation triggers rate hikes which '
                                 'reduce credit (Bernanke & Gertler 1995; Mishkin 1996)',
        },
        {
            'source': 'gdp_growth', 'mediator': 'unemployment',
            'target': 'govt_consumption',
            'type': 'mediating', 'expected_sign': +1, 'strength': 'weak',
            'theoretical_basis': 'Automatic stabilisers — GDP affects spending through '
                                 'unemployment insurance and tax revenue '
                                 '(Blanchard Ch.22; Auerbach & Gorodnichenko 2012)',
        },

        # ==============================================================
        # SYNERGISTIC — interaction/moderation effects X*Z -> Y
        # ==============================================================
        {
            'source': 'private_credit', 'moderator': 'electricity_access',
            'target': 'gdp_growth',
            'type': 'synergistic', 'expected_sign': +1, 'strength': 'moderate',
            'theoretical_basis': 'Credit x infrastructure complementarity — credit is more '
                                 'productive where infrastructure exists '
                                 '(Sahay et al. 2015; Demirguc-Kunt & Levine 2008)',
        },
        {
            'source': 'school_enrollment', 'moderator': 'electricity_access',
            'target': 'gdp_growth',
            'type': 'synergistic', 'expected_sign': +1, 'strength': 'moderate',
            'theoretical_basis': 'Human capital x infrastructure — educated workers more '
                                 'productive with electricity/internet access '
                                 '(Hanushek & Woessmann 2012; World Bank 2019)',
        },

        # ==============================================================
        # STRUCTURAL — regime-dependent distributional change
        # ==============================================================
        {
            'source': 'inflation_cpi', 'target': 'inflation_cpi',
            'type': 'structural', 'expected_sign': 0, 'strength': 'moderate',
            'theoretical_basis': 'Inflation targeting regime shift — Kenya adopted IT framework '
                                 'in 2011; distribution of inflation changed '
                                 '(CBK Monetary Policy Statement; IMF Art. IV 2012)',
        },

        # ==============================================================
        # FUNCTIONAL — nonlinear f(X) ~ Y
        # ==============================================================
        {
            'source': 'gdp_growth', 'target': 'life_expectancy',
            'type': 'functional', 'expected_sign': +1, 'strength': 'strong',
            'theoretical_basis': 'Preston Curve — log-linear relationship between income '
                                 'and life expectancy (Preston 1975; Deaton 2013)',
        },
    ]


def get_known_null_relationships() -> list[dict]:
    """Variable pairs where economic theory asserts no relationship exists."""
    return [
        {
            'source': 'life_expectancy', 'target': 'real_interest_rate',
            'reason': 'No established macro transmission between demographic health and '
                      'monetary policy rate at annual frequency',
        },
        {
            'source': 'school_enrollment', 'target': 'current_account',
            'reason': 'Education enrollment does not affect trade balance at annual horizon; '
                      'any correlation is spurious through GDP',
        },
        {
            'source': 'mobile_subscriptions', 'target': 'real_interest_rate',
            'reason': 'Telecoms adoption is not a monetary variable; no transmission channel',
        },
        {
            'source': 'urban_population', 'target': 'inflation_cpi',
            'reason': 'Urbanisation is a slow structural trend; no year-to-year causal link '
                      'to inflation at annual frequency',
        },
    ]


def get_ground_truth_by_type() -> dict[str, list[dict]]:
    """Group ground truth by relationship type."""
    gt = get_typed_ground_truth()
    by_type: dict[str, list[dict]] = {}
    for entry in gt:
        t = entry['type']
        if t not in by_type:
            by_type[t] = []
        by_type[t].append(entry)
    return by_type


def get_all_gt_variables() -> set[str]:
    """Return every variable name that appears anywhere in the ground truth."""
    gt = get_typed_ground_truth()
    vars_: set[str] = set()
    for e in gt:
        vars_.add(e['source'])
        vars_.add(e['target'])
        if 'mediator' in e:
            vars_.add(e['mediator'])
        if 'moderator' in e:
            vars_.add(e['moderator'])
    return vars_


# ---------------------------------------------------------------------------
# Quick summary for verification
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    gt = get_typed_ground_truth()
    by_type = get_ground_truth_by_type()
    nulls = get_known_null_relationships()
    print(f"Total GT relationships: {len(gt)}")
    print(f"Null pairs: {len(nulls)}")
    print("\nCounts by type:")
    for t, rels in sorted(by_type.items()):
        print(f"  {t:15s}: {len(rels)}")
    print(f"\nAll GT variables ({len(get_all_gt_variables())}):")
    print(" ", sorted(get_all_gt_variables()))

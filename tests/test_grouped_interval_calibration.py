"""Staged numerical/group metadata regressions; no FPL training or evaluation."""
from dataclasses import FrozenInstanceError
import inspect

import numpy as np
import pytest

from fpl.model import grouped_interval_calibration as module
Cell=module.CalibrationCell
Calibration=module.GroupedCalibration
fit=module.fit_grouped_calibration
apply=module.apply_grouped_calibration


def calibration():
    return fit([0,1,2,3,4,5],[0]*6,[4]*6,['MID']*6,['large']*4+['sparse']*2,alpha=.2,min_group_size=4)


def test_exact_signed_scores_rank_and_all_counts_retained():
    fitted=calibration()
    assert fitted.cells==(Cell('MID',None,6,1.),Cell('MID','large',4,0.))
    assert fitted.group_counts==(('MID','large',4),('MID','sparse',2))
    assert fitted.alpha==.2 and fitted.min_group_size==4


def test_application_group_sparse_and_unseen_fallback_stays_aligned():
    result=apply(calibration(),[0,0,0],[2,2,2],[4,4,4],['MID']*3,['sparse','large','new'])
    np.testing.assert_array_equal(result[0],[-1,0,-1])
    np.testing.assert_array_equal(result[1],[5,4,5])
    np.testing.assert_array_equal(result[2],[1,0,1])
    np.testing.assert_array_equal(result[3],['position','group','position'])


def test_negative_shrink_uses_existing_median_bracketing():
    fitted=fit([2]*4,[0]*4,[4]*4,['MID']*4,['one']*4,alpha=.2,min_group_size=4)
    assert fitted.cells[0].adjustment==-2
    lower,upper,_,sources=apply(fitted,[0,0],[.5,3.5],[4,4],['MID','MID'],['one','new'])
    np.testing.assert_array_equal(lower,[.5,2])
    np.testing.assert_array_equal(upper,[2,3.5])
    np.testing.assert_array_equal(sources,['group','position'])


def test_positions_use_separate_calibration_and_deterministic_cells():
    fitted=fit([2]*4+[8]*4,[0]*8,[4]*8,['MID']*4+['DEF']*4,['one']*8,alpha=.2,min_group_size=4)
    assert [(cell.position,cell.group) for cell in fitted.cells]==[('DEF',None),('DEF','one'),('MID',None),('MID','one')]
    assert fitted.group_counts==(('DEF','one',4),('MID','one',4))
    _,_,adjustments,sources=apply(fitted,[0,0],[2,2],[4,4],['MID','DEF'],['new','new'])
    np.testing.assert_array_equal(adjustments,[-2,4])
    np.testing.assert_array_equal(sources,['position','position'])


def test_insufficient_eligible_group_rank_raises_instead_of_falling_back():
    with pytest.raises(ValueError,match='insufficient calibration sample'):
        fit([2]*6,[0]*6,[4]*6,['MID']*6,['small']*2+['large']*4,alpha=.2,min_group_size=2)


def test_insufficient_position_fallback_rank_raises():
    with pytest.raises(ValueError,match='insufficient calibration sample'):
        fit([2,2],[0,0],[4,4],['MID','MID'],['one','one'],alpha=.2,min_group_size=99)


@pytest.mark.parametrize('min_size',[0,-1,True,1.,np.int64(4)])
def test_min_size_requires_positive_python_int(min_size):
    with pytest.raises(ValueError,match='positive Python integer'):
        fit([2]*4,[0]*4,[4]*4,['MID']*4,['one']*4,alpha=.2,min_group_size=min_size)


@pytest.mark.parametrize('alpha',[0,1,-1,float('nan'),float('inf'),True,'0.2'])
def test_invalid_alpha_refused(alpha):
    with pytest.raises(ValueError):
        fit([2]*4,[0]*4,[4]*4,['MID']*4,['one']*4,alpha=alpha,min_group_size=4)


@pytest.mark.parametrize('y,lower,upper,positions,groups',[
    ([],[],[],[],[]),
    ([2]*4,[0]*3,[4]*4,['MID']*4,['one']*4),
    ([2,2,float('nan'),2],[0]*4,[4]*4,['MID']*4,['one']*4),
    ([2]*4,[5]*4,[4]*4,['MID']*4,['one']*4),
    ([2]*4,[0]*4,[4]*4,['MID']*3,['one']*4),
    ([2]*4,[0]*4,[4]*4,['MID']*4,['one']*3),
    ([2]*4,[0]*4,[4]*4,['MID']*3+[''],['one']*4),
    ([2]*4,[0]*4,[4]*4,['MID']*4,['one']*3+[None]),
    ([2]*4,[0]*4,[4]*4,['MID']*3+[1],['one']*4),
    ([2]*4,[0]*4,[4]*4,[['MID']]*4,['one']*4),
])
def test_fit_alignment_numeric_bounds_and_labels_refused(y,lower,upper,positions,groups):
    with pytest.raises(ValueError):fit(y,lower,upper,positions,groups,alpha=.2,min_group_size=4)


@pytest.mark.parametrize('lower,median,upper,positions,groups',[
    ([0],[5],[4],['MID'],['one']),([5],[3],[4],['MID'],['one']),
    ([0],[2],[float('inf')],['MID'],['one']),([0],[2],[4],[],['one']),
    ([0],[2],[4],['MID'],['']),([0],[2],[4],['FWD'],['one']),
])
def test_application_refuses_invalid_median_bounds_labels_or_unknown_position(lower,median,upper,positions,groups):
    with pytest.raises(ValueError):apply(calibration(),lower,median,upper,positions,groups)


def test_application_signature_has_no_outcomes_and_fit_does_not_retain_targets():
    y=np.array([0.,1.,2.,3.,4.,5.])
    fitted=fit(y,[0]*6,[4]*6,['MID']*6,['large']*4+['sparse']*2,alpha=.2,min_group_size=4)
    first=apply(fitted,[0],[2],[4],['MID'],['new'])
    y[:]=99999
    second=apply(fitted,[0],[2],[4],['MID'],['new'])
    for a,b in zip(first,second):np.testing.assert_array_equal(a,b)
    assert 'y' not in inspect.signature(apply).parameters
    with pytest.raises(TypeError):apply(fitted,[0],[2],[4],['MID'],['new'],y=[9999])


def test_cells_and_counts_are_immutable():
    fitted=calibration()
    with pytest.raises(FrozenInstanceError):fitted.alpha=.5
    with pytest.raises(FrozenInstanceError):fitted.cells[0].n=1
    assert isinstance(fitted.cells,tuple) and isinstance(fitted.group_counts,tuple)


@pytest.mark.parametrize('position,group,n,adjustment',[
    ('',None,4,1),('MID','',4,1),('MID',1,4,1),('MID',None,True,1),
    ('MID',None,0,1),('MID',None,4,float('nan')),('MID',None,4,True),
])
def test_cell_validates_labels_counts_and_adjustments(position,group,n,adjustment):
    with pytest.raises(ValueError):Cell(position,group,n,adjustment)


@pytest.mark.parametrize('cells,counts',[
    ([Cell('MID',None,4,1)],(('MID','one',4),)),
    ((Cell('MID',None,4,1),Cell('MID',None,4,1)),(('MID','one',4),)),
    ((Cell('MID','one',4,1),),(('MID','one',4),)),
    ((Cell('MID',None,4,1),),(('MID','one',True),)),
    ((Cell('MID',None,4,1),),(('MID','one',3),)),
    ((Cell('MID',None,4,1),),(('MID','one',2),('MID','one',2))),
    ((Cell('MID',None,4,1),),(('DEF','one',4),)),
    ((Cell('MID',None,4,1),),(('MID','one',4),)),
    ((Cell('MID',None,4,1),Cell('MID','one',3,1)),(('MID','one',4),)),
    ((Cell('MID',None,4,1),Cell('MID','one',2,1)),(('MID','one',2),('MID','two',2))),
    ((Cell('MID',None,4,1),),[['MID','one',4]]),
])
def test_calibration_metadata_duplicates_partition_minimum_and_immutable_schema(cells,counts):
    with pytest.raises(ValueError):Calibration(.2,4,cells,counts)


def test_duplicate_group_cells_refused():
    with pytest.raises(ValueError,match='duplicate calibration'):
        Calibration(.2,4,(Cell('MID',None,4,1),Cell('MID','one',4,1),Cell('MID','one',4,1)),(('MID','one',4),))


def test_unsorted_group_counts_refused():
    with pytest.raises(ValueError,match='sorted by position/group'):
        Calibration(.2,4,(Cell('MID',None,4,1),),(('MID','z',2),('MID','a',2)))


def test_unhashable_label_is_explicit_validation_error():
    labels=np.empty(4,dtype=object);labels[:]=['one','one','one',['nested']]
    with pytest.raises(ValueError,match='nonempty text labels'):
        fit([2]*4,[0]*4,[4]*4,['MID']*4,labels,alpha=.2,min_group_size=4)

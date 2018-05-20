DUMMY_VARIABLES_T_F = ['school_charter', 
'school_magnet', 
'school_year_round', 
'school_nlns', 
'school_kipp', 
'school_charter_ready_promise', 
'teacher_teach_for_america',
'teacher_ny_teaching_fellow',  
'eligible_double_your_impact_match',
'eligible_almost_home_match']


DUMMY_VARIABLES_NOT_T_F = ['fulfillment_labor_materials']

CATEGORICAL_VARIABLES = ['school_city', 
'school_state', 
'school_zip', 
'school_metro',
'school_district', 
'school_county', 
'teacher_prefix',
'primary_focus_subject', 
'primary_focus_area', 
'secondary_focus_subject',
'secondary_focus_area', 
'resource_type', 
'poverty_level', 
'grade_level']

CONTINUOUS_VARIABLES = ['total_price_excluding_optional_support', 
'total_price_including_optional_support',
'students_reached']

DATE_VARIABLE = ['date_posted']

ID_VARIABLES = ['teacher_acctid', 
'schoolid', 
'school_ncesid']

GEO_VARIABLES = ['school_latitude', 
'school_longitude']

TARGET_VARIABLES = ['fully_funded']

IDX = ['projectid']


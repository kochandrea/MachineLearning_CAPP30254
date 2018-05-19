ALREADY_DUMMY_VARIABLES = ['school_charter', 
'school_magnet', 
'school_year_round', 
'school_nlns', 
'school_kipp', 
'school_charter_ready_promise', 
'teacher_teach_for_america',
'teacher_ny_teaching_fellow', 
'fulfillment_labor_materials', 
'eligible_double_your_impact_match',
'eligible_almost_home_match']

TO_BE_DUMMYTIZED_VARIABLES = ['school_city', 
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

TO_BE_DISCRETIZED_VARIABLES = ['total_price_excluding_optional_support', 
'total_price_including_optional_support',
'students_reached']

DATE_VARIABLE = ['date_posted']

ID_VARIABLES = ['teacher_acctid', 
'schoolid', 
'school_ncesid']

GEO_VARIABLES = ['school_latitude', 
'school_longitude']


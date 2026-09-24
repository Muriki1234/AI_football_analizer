[2026-09-17 16:37:55] [SYSTEM] Memory system initialized.
[2026-09-17 22:37:17] [session_night_20260917] [IDLE] [EVENT=SESSION_START] session=session_night_20260917 mainline=18_zone_radar secondary=none
[2026-09-17 23:36:23] [session_night_20260917] [IDLE] [EVENT=NOTE] [EVENT=ACTION_COMPLETED] Verified bug fix for TelestrationCanvas infinite loop OOM
[2026-09-17 23:37:55] [session_night_20260917] [GRAPH] [EVENT=OPPORTUNITY_TRANSITION] opportunity=18_zone_radar from=ACTIVATED to=EXPLORING
[2026-09-17 23:38:00] [session_night_20260917] [GRAPH] [EVENT=BRANCH_DEACTIVATED] tier=main opportunity=18_zone_radar status=VALIDATED
[2026-09-17 23:38:00] [session_night_20260917] [GRAPH] [EVENT=OPPORTUNITY_TRANSITION] opportunity=18_zone_radar from=EXPLORING to=VALIDATED
[2026-09-17 23:38:29] [session_night_20260917] [GRAPH] [EVENT=BRANCH_SWITCHED] tier=main from=18_zone_radar to=vertical_crop_916
[2026-09-17 23:38:29] [session_night_20260917] [GRAPH] [EVENT=OPPORTUNITY_TRANSITION] opportunity=vertical_crop_916 from=PARKED to=ACTIVATED
[2026-09-17 23:38:34] [session_night_20260917] [GRAPH] [EVENT=OPPORTUNITY_TRANSITION] opportunity=vertical_crop_916 from=ACTIVATED to=EXPLORING
[2026-09-17 23:38:38] [session_night_20260917] [EXPLORING] [EVENT=CHECKPOINT] current_action=authoring_and_verifying_vertical_crop_tests next_action=validate_and_benchmark
[2026-09-17 23:39:10] [session_night_20260917] [GRAPH] [EVENT=BRANCH_DEACTIVATED] tier=main opportunity=vertical_crop_916 status=VALIDATED
[2026-09-17 23:39:10] [session_night_20260917] [GRAPH] [EVENT=OPPORTUNITY_TRANSITION] opportunity=vertical_crop_916 from=EXPLORING to=VALIDATED
[2026-09-17 23:39:35] [session_night_20260917] [GRAPH] [EVENT=OPPORTUNITY_CREATED] opportunity=jersey_tracklet_voting status=PARKED
[2026-09-17 23:39:55] [session_night_20260917] [GRAPH] [EVENT=BRANCH_SWITCHED] tier=main from=vertical_crop_916 to=jersey_tracklet_voting
[2026-09-17 23:39:55] [session_night_20260917] [GRAPH] [EVENT=OPPORTUNITY_TRANSITION] opportunity=jersey_tracklet_voting from=PARKED to=ACTIVATED
[2026-09-17 23:40:01] [session_night_20260917] [GRAPH] [EVENT=OPPORTUNITY_TRANSITION] opportunity=jersey_tracklet_voting from=ACTIVATED to=EXPLORING
[2026-09-17 23:40:06] [session_night_20260917] [EXPLORING] [EVENT=CHECKPOINT] current_action=building_jersey_voting_engine next_action=run_unit_tests
[2026-09-17 23:40:51] [session_night_20260917] [GRAPH] [EVENT=BRANCH_DEACTIVATED] tier=main opportunity=jersey_tracklet_voting status=VALIDATED
[2026-09-17 23:40:51] [session_night_20260917] [GRAPH] [EVENT=OPPORTUNITY_TRANSITION] opportunity=jersey_tracklet_voting from=EXPLORING to=VALIDATED
[2026-09-17 23:41:17] [session_night_20260917] [GRAPH] [EVENT=OPPORTUNITY_CREATED] opportunity=pitch_control_voronoi status=PARKED
[2026-09-17 23:41:22] [session_night_20260917] [GRAPH] [EVENT=BRANCH_SWITCHED] tier=main from=jersey_tracklet_voting to=pitch_control_voronoi
[2026-09-17 23:41:22] [session_night_20260917] [GRAPH] [EVENT=OPPORTUNITY_TRANSITION] opportunity=pitch_control_voronoi from=PARKED to=ACTIVATED
[2026-09-17 23:41:27] [session_night_20260917] [GRAPH] [EVENT=OPPORTUNITY_TRANSITION] opportunity=pitch_control_voronoi from=ACTIVATED to=EXPLORING
[2026-09-17 23:41:33] [session_night_20260917] [EXPLORING] [EVENT=CHECKPOINT] current_action=authoring_pitch_control_algorithm next_action=run_physics_and_grid_unittests
[2026-09-17 23:42:23] [session_night_20260917] [GRAPH] [EVENT=BRANCH_DEACTIVATED] tier=main opportunity=pitch_control_voronoi status=VALIDATED
[2026-09-17 23:42:23] [session_night_20260917] [GRAPH] [EVENT=OPPORTUNITY_TRANSITION] opportunity=pitch_control_voronoi from=EXPLORING to=VALIDATED
[2026-09-17 23:42:40] [session_night_20260917] [IDLE] [EVENT=CHECKPOINT] current_action=overnight_rd_completed_all_prototypes_validated next_action=morning_intelligence_report_generated
[2026-09-19 16:26:55] [session_night_20260919] [IDLE] [EVENT=SESSION_START] session=session_night_20260919 mainline=none secondary=none
[2026-09-19 16:27:01] [session_night_20260919] [RECON] [EVENT=CHECKPOINT] Initiating session_night_20260919: profiling long-video detection performance bottlenecks
[2026-09-19 16:29:09] [session_night_20260919] [GRAPH] [EVENT=OPPORTUNITY_CREATED] opportunity=long_video_accel status=PARKED
[2026-09-19 16:29:15] [session_night_20260919] [GRAPH] [EVENT=BRANCH_ACTIVATED] tier=main opportunity=long_video_accel
[2026-09-19 16:29:15] [session_night_20260919] [GRAPH] [EVENT=OPPORTUNITY_TRANSITION] opportunity=long_video_accel from=PARKED to=ACTIVATED
[2026-09-19 16:29:21] [session_night_20260919] [GRAPH] [EVENT=OPPORTUNITY_TRANSITION] opportunity=long_video_accel from=ACTIVATED to=EXPLORING
[2026-09-19 16:29:27] [session_night_20260919] [EXPLORING] [EVENT=CHECKPOINT] Developing detection acceleration engine with async prefetch, tactical view gating, and adaptive stride
[2026-09-19 16:32:32] [session_night_20260919] [GRAPH] [EVENT=BRANCH_DEACTIVATED] tier=main opportunity=long_video_accel status=VALIDATED
[2026-09-19 16:32:32] [session_night_20260919] [GRAPH] [EVENT=OPPORTUNITY_TRANSITION] opportunity=long_video_accel from=EXPLORING to=VALIDATED
[2026-09-19 16:33:05] [session_night_20260919] [VALIDATING] [EVENT=CHECKPOINT] 13 unittests passed; 66.7% compute reduction verified
[2026-09-19 18:48:03] [test_session] [VALIDATING] [EVENT=SESSION_START] session=test_session overnight=True budget=120m mainline=none secondary=none
[2026-09-19 18:48:13] [test_session] [VALIDATING] [EVENT=SESSION_END] Overnight mode ended explicitly.
[2026-09-19 18:48:55] [session_night_20260919] [GRAPH] [EVENT=EXTERNAL_RESEARCH] opportunity=long_video_accel decision=hybrid sources=2
[2026-09-19 18:50:51] [session_night_20260919] [GRAPH] [EVENT=EXTERNAL_RESEARCH] opportunity=sample_op decision=hybrid sources=2
[2026-09-19 19:46:46] [session_night_20260920] [VALIDATING] [EVENT=SESSION_START] session=session_night_20260920 overnight=True budget=240m mainline=none secondary=none
[2026-09-19 19:47:23] [session_night_20260920] [GRAPH] [EVENT=OPPORTUNITY_CREATED] opportunity=pitch_homography_stabilizer status=PARKED
[2026-09-19 19:47:29] [session_night_20260920] [GRAPH] [EVENT=BRANCH_ACTIVATED] tier=main opportunity=pitch_homography_stabilizer
[2026-09-19 19:47:29] [session_night_20260920] [GRAPH] [EVENT=OPPORTUNITY_TRANSITION] opportunity=pitch_homography_stabilizer from=PARKED to=ACTIVATED
[2026-09-19 19:47:35] [session_night_20260920] [GRAPH] [EVENT=OPPORTUNITY_TRANSITION] opportunity=pitch_homography_stabilizer from=ACTIVATED to=EXPLORING
[2026-09-19 19:47:43] [session_night_20260920] [EXPLORING] [EVENT=CHECKPOINT] Investigating pitch keypoint drift, homography failure modes, and external SOTA
[2026-09-19 19:48:34] [session_night_20260920] [GRAPH] [EVENT=EXTERNAL_RESEARCH] opportunity=pitch_homography_stabilizer decision=hybrid sources=3
[2026-09-19 19:50:37] [session_night_20260920] [GRAPH] [EVENT=BRANCH_DEACTIVATED] tier=main opportunity=pitch_homography_stabilizer status=VALIDATED
[2026-09-19 19:50:37] [session_night_20260920] [GRAPH] [EVENT=OPPORTUNITY_TRANSITION] opportunity=pitch_homography_stabilizer from=EXPLORING to=VALIDATED
[2026-09-19 19:53:36] [session_night_20260920] [GRAPH] [EVENT=OPPORTUNITY_CREATED] opportunity=heatmap_spatial_stabilizer status=PARKED
[2026-09-19 19:53:56] [session_night_20260920] [GRAPH] [EVENT=BRANCH_SWITCHED] tier=main from=pitch_homography_stabilizer to=heatmap_spatial_stabilizer
[2026-09-19 19:53:56] [session_night_20260920] [GRAPH] [EVENT=OPPORTUNITY_TRANSITION] opportunity=heatmap_spatial_stabilizer from=PARKED to=ACTIVATED
[2026-09-19 19:54:05] [session_night_20260920] [GRAPH] [EVENT=EXTERNAL_RESEARCH] opportunity=heatmap_spatial_stabilizer decision=adopted sources=3
[2026-09-19 19:54:10] [session_night_20260920] [GRAPH] [EVENT=OPPORTUNITY_TRANSITION] opportunity=heatmap_spatial_stabilizer from=ACTIVATED to=EXPLORING
[2026-09-19 19:57:27] [session_night_20260920] [GRAPH] [EVENT=BRANCH_DEACTIVATED] tier=main opportunity=heatmap_spatial_stabilizer status=VALIDATED
[2026-09-19 19:57:27] [session_night_20260920] [GRAPH] [EVENT=OPPORTUNITY_TRANSITION] opportunity=heatmap_spatial_stabilizer from=EXPLORING to=VALIDATED
[2026-09-19 19:58:16] [session_night_20260920] [GRAPH] [EVENT=OPPORTUNITY_CREATED] opportunity=pipeline_db_debouncer status=PARKED
[2026-09-19 19:58:20] [session_night_20260920] [GRAPH] [EVENT=BRANCH_ACTIVATED] tier=secondary opportunity=pipeline_db_debouncer
[2026-09-19 19:58:20] [session_night_20260920] [GRAPH] [EVENT=OPPORTUNITY_TRANSITION] opportunity=pipeline_db_debouncer from=PARKED to=ACTIVATED
[2026-09-19 19:58:49] [session_night_20260920] [GRAPH] [EVENT=EXTERNAL_RESEARCH] opportunity=pipeline_db_debouncer decision=adopted sources=2
[2026-09-19 19:58:54] [session_night_20260920] [GRAPH] [EVENT=OPPORTUNITY_TRANSITION] opportunity=pipeline_db_debouncer from=ACTIVATED to=EXPLORING
[2026-09-19 19:59:25] [session_night_20260920] [GRAPH] [EVENT=BRANCH_DEACTIVATED] tier=secondary opportunity=pipeline_db_debouncer status=VALIDATED
[2026-09-19 19:59:25] [session_night_20260920] [GRAPH] [EVENT=OPPORTUNITY_TRANSITION] opportunity=pipeline_db_debouncer from=EXPLORING to=VALIDATED
[2026-09-19 20:00:23] [session_night_20260920] [GRAPH] [EVENT=EXTERNAL_RESEARCH] opportunity=sample_op decision=hybrid sources=2
[2026-09-19 20:01:17] [session_night_20260920] [COMPLETE] [EVENT=CHECKPOINT] Completed all 8/8 validated opportunities
[2026-09-19 20:01:21] [session_night_20260920] [COMPLETE] [EVENT=SESSION_END] Overnight mode ended explicitly.
[2026-09-19 20:06:01] [session_night_20260920] [GRAPH] [EVENT=EXTERNAL_RESEARCH] opportunity=long_video_accel decision=adopted sources=2
[2026-09-19 22:55:08] [test_sess] [RECON] [EVENT=SESSION_END] Overnight mode ended explicitly.
[2026-09-19 22:55:08] [test_sess] [RECON] [EVENT=SESSION_END] Overnight mode ended explicitly.
[2026-09-19 22:55:08] [test_sess] [COMPLETE] [EVENT=CHECKPOINT] current_action=all_opportunities_validated next_action=morning_report_delivered
[2026-09-19 22:55:08] [test_sess] [DISCOVERY] [EVENT=CHECKPOINT] [⚠️ [GUARD_INTERCEPT] Intercepted attempt to set phase='complete' while overnight budget remains (0.0m / 240.0m). Mandating Rediscovery Phase instead of early exit.]
[2026-09-19 22:55:08] [session_night_20260920] [GRAPH] [EVENT=EXTERNAL_RESEARCH] opportunity=sample_op decision=hybrid sources=2
[2026-09-19 22:55:38] [session_night_20260920] [COMPLETE] [EVENT=SESSION_START] session=session_night_20260920 overnight=True budget=240m mainline=none secondary=none
[2026-09-19 22:55:48] [session_night_20260920] [DISCOVERY] [EVENT=CHECKPOINT] Entering Mandatory Rediscovery Phase: deep inspection of server/handler.py, pipeline/tasks.py, and production logs
[2026-09-19 22:58:06] [session_night_20260920] [GRAPH] [EVENT=OPPORTUNITY_CREATED] opportunity=ball_action_spotting status=PARKED
[2026-09-19 22:58:12] [session_night_20260920] [GRAPH] [EVENT=OPPORTUNITY_CREATED] opportunity=pipeline_concurrency_scheduler status=PARKED
[2026-09-19 22:58:15] [session_night_20260920] [GRAPH] [EVENT=BRANCH_SWITCHED] tier=main from=heatmap_spatial_stabilizer to=ball_action_spotting
[2026-09-19 22:58:15] [session_night_20260920] [GRAPH] [EVENT=OPPORTUNITY_TRANSITION] opportunity=ball_action_spotting from=PARKED to=ACTIVATED
[2026-09-19 22:58:20] [session_night_20260920] [GRAPH] [EVENT=OPPORTUNITY_TRANSITION] opportunity=ball_action_spotting from=ACTIVATED to=EXPLORING
[2026-09-19 22:58:25] [session_night_20260920] [DISCOVERY] [EVENT=CHECKPOINT] Researching SoccerNet FOOTPASS and Kloppy event spotting models; authoring pass_event_detector.py
[2026-09-19 22:58:32] [session_night_20260920] [GRAPH] [EVENT=EXTERNAL_RESEARCH] opportunity=ball_action_spotting decision=hybrid sources=3
[2026-09-19 23:01:02] [session_night_20260920] [GRAPH] [EVENT=BRANCH_DEACTIVATED] tier=main opportunity=ball_action_spotting status=VALIDATED
[2026-09-19 23:01:02] [session_night_20260920] [GRAPH] [EVENT=OPPORTUNITY_TRANSITION] opportunity=ball_action_spotting from=EXPLORING to=VALIDATED
[2026-09-19 23:01:12] [session_night_20260920] [GRAPH] [EVENT=BRANCH_SWITCHED] tier=main from=ball_action_spotting to=pipeline_concurrency_scheduler
[2026-09-19 23:01:12] [session_night_20260920] [GRAPH] [EVENT=OPPORTUNITY_TRANSITION] opportunity=pipeline_concurrency_scheduler from=PARKED to=ACTIVATED
[2026-09-19 23:01:18] [session_night_20260920] [GRAPH] [EVENT=OPPORTUNITY_TRANSITION] opportunity=pipeline_concurrency_scheduler from=ACTIVATED to=EXPLORING
[2026-09-19 23:01:23] [session_night_20260920] [DISCOVERY] [EVENT=CHECKPOINT] Researching multi-process CUDA concurrency, thread synchronization events, and VRAM guardrails; authoring pipeline_concurrency_scheduler.py
[2026-09-19 23:01:42] [session_night_20260920] [GRAPH] [EVENT=EXTERNAL_RESEARCH] opportunity=pipeline_concurrency_scheduler decision=hybrid sources=2
[2026-09-19 23:02:53] [test_sess] [RECON] [EVENT=SESSION_END] Overnight mode ended explicitly.
[2026-09-19 23:02:53] [test_sess] [RECON] [EVENT=SESSION_END] Overnight mode ended explicitly.
[2026-09-19 23:02:53] [test_sess] [COMPLETE] [EVENT=CHECKPOINT] current_action=all_opportunities_validated next_action=morning_report_delivered
[2026-09-19 23:02:53] [test_sess] [DISCOVERY] [EVENT=CHECKPOINT] [⚠️ [GUARD_INTERCEPT] Intercepted attempt to set phase='complete' while overnight budget remains (0.0m / 240.0m). Mandating Rediscovery Phase instead of early exit.]
[2026-09-19 23:02:53] [session_night_20260920] [GRAPH] [EVENT=EXTERNAL_RESEARCH] opportunity=sample_op decision=hybrid sources=2
[2026-09-19 23:02:59] [session_night_20260920] [GRAPH] [EVENT=BRANCH_DEACTIVATED] tier=main opportunity=pipeline_concurrency_scheduler status=VALIDATED
[2026-09-19 23:02:59] [session_night_20260920] [GRAPH] [EVENT=OPPORTUNITY_TRANSITION] opportunity=pipeline_concurrency_scheduler from=EXPLORING to=VALIDATED
[2026-09-19 23:08:36] [session_night_20260920] [GRAPH] [EVENT=OPPORTUNITY_CREATED] opportunity=speed_kinematics_filter status=PARKED
[2026-09-19 23:08:41] [session_night_20260920] [GRAPH] [EVENT=OPPORTUNITY_CREATED] opportunity=tactical_passing_coach status=PARKED
[2026-09-19 23:08:49] [session_night_20260920] [GRAPH] [EVENT=BRANCH_SWITCHED] tier=main from=pipeline_concurrency_scheduler to=speed_kinematics_filter
[2026-09-19 23:08:49] [session_night_20260920] [GRAPH] [EVENT=OPPORTUNITY_TRANSITION] opportunity=speed_kinematics_filter from=PARKED to=ACTIVATED
[2026-09-19 23:08:54] [session_night_20260920] [GRAPH] [EVENT=BRANCH_SWITCHED] tier=secondary from=pipeline_db_debouncer to=tactical_passing_coach
[2026-09-19 23:08:54] [session_night_20260920] [GRAPH] [EVENT=OPPORTUNITY_TRANSITION] opportunity=tactical_passing_coach from=PARKED to=ACTIVATED
[2026-09-19 23:09:00] [session_night_20260920] [GRAPH] [EVENT=OPPORTUNITY_TRANSITION] opportunity=speed_kinematics_filter from=ACTIVATED to=EXPLORING
[2026-09-19 23:09:07] [session_night_20260920] [DISCOVERY] [EVENT=CHECKPOINT] Researching kinematic trajectory filtering (Savitzky-Golay, STATSports/Catapult speed deadband) and authoring speed_kinematics_filter.py
[2026-09-19 23:09:30] [session_night_20260920] [GRAPH] [EVENT=EXTERNAL_RESEARCH] opportunity=speed_kinematics_filter decision=hybrid sources=2
[2026-09-19 23:17:03] [session_night_20260920] [GRAPH] [EVENT=BRANCH_DEACTIVATED] tier=main opportunity=speed_kinematics_filter status=VALIDATED
[2026-09-19 23:17:03] [session_night_20260920] [GRAPH] [EVENT=OPPORTUNITY_TRANSITION] opportunity=speed_kinematics_filter from=EXPLORING to=VALIDATED
[2026-09-19 23:17:19] [session_night_20260920] [GRAPH] [EVENT=OPPORTUNITY_TRANSITION] opportunity=tactical_passing_coach from=ACTIVATED to=EXPLORING
[2026-09-19 23:17:28] [session_night_20260920] [EXECUTE] [EVENT=CHECKPOINT] Executing Opportunity 12: Tactical Passing Coach
[2026-09-19 23:18:51] [session_night_20260920] [GRAPH] [EVENT=EXTERNAL_RESEARCH] opportunity=tactical_passing_coach decision=adapted sources=3
[2026-09-19 23:24:33] [session_night_20260920] [GRAPH] [EVENT=BRANCH_DEACTIVATED] tier=main opportunity=tactical_passing_coach status=VALIDATED
[2026-09-19 23:24:33] [session_night_20260920] [GRAPH] [EVENT=OPPORTUNITY_TRANSITION] opportunity=tactical_passing_coach from=EXPLORING to=VALIDATED
[2026-09-19 23:24:52] [session_night_20260920] [DISCOVERY] [EVENT=CHECKPOINT] Triggering Mandatory Rediscovery Phase after Opportunity 12 validation. Scanning codebase & production logs.
[2026-09-19 23:26:34] [session_night_20260920] [GRAPH] [EVENT=OPPORTUNITY_CREATED] opportunity=camera_motion_ransac status=PARKED
[2026-09-19 23:26:39] [session_night_20260920] [GRAPH] [EVENT=OPPORTUNITY_CREATED] opportunity=jersey_vision_integrator status=PARKED
[2026-09-19 23:26:45] [session_night_20260920] [GRAPH] [EVENT=BRANCH_SWITCHED] tier=main from=speed_kinematics_filter to=camera_motion_ransac
[2026-09-19 23:26:45] [session_night_20260920] [GRAPH] [EVENT=OPPORTUNITY_TRANSITION] opportunity=camera_motion_ransac from=PARKED to=ACTIVATED
[2026-09-19 23:26:50] [session_night_20260920] [GRAPH] [EVENT=BRANCH_SWITCHED] tier=secondary from=tactical_passing_coach to=jersey_vision_integrator
[2026-09-19 23:26:50] [session_night_20260920] [GRAPH] [EVENT=OPPORTUNITY_TRANSITION] opportunity=jersey_vision_integrator from=PARKED to=ACTIVATED
[2026-09-19 23:27:03] [session_night_20260920] [GRAPH] [EVENT=OPPORTUNITY_TRANSITION] opportunity=camera_motion_ransac from=ACTIVATED to=EXPLORING
[2026-09-19 23:27:09] [session_night_20260920] [EXECUTE] [EVENT=CHECKPOINT] Executing Opportunity 13: Robust RANSAC Camera Motion Estimator
[2026-09-19 23:27:18] [session_night_20260920] [GRAPH] [EVENT=EXTERNAL_RESEARCH] opportunity=camera_motion_ransac decision=adapted sources=2
[2026-09-19 23:30:58] [session_night_20260920] [GRAPH] [EVENT=BRANCH_DEACTIVATED] tier=main opportunity=camera_motion_ransac status=VALIDATED
[2026-09-19 23:30:58] [session_night_20260920] [GRAPH] [EVENT=OPPORTUNITY_TRANSITION] opportunity=camera_motion_ransac from=EXPLORING to=VALIDATED
[2026-09-19 23:31:10] [session_night_20260920] [GRAPH] [EVENT=OPPORTUNITY_TRANSITION] opportunity=jersey_vision_integrator from=ACTIVATED to=EXPLORING
[2026-09-19 23:31:16] [session_night_20260920] [EXECUTE] [EVENT=CHECKPOINT] Executing Opportunity 14: Jersey Vision & Pipeline Integrator
[2026-09-19 23:31:24] [session_night_20260920] [GRAPH] [EVENT=EXTERNAL_RESEARCH] opportunity=jersey_vision_integrator decision=adapted sources=3
[2026-09-19 23:34:04] [session_night_20260920] [GRAPH] [EVENT=BRANCH_DEACTIVATED] tier=main opportunity=jersey_vision_integrator status=VALIDATED
[2026-09-19 23:34:04] [session_night_20260920] [GRAPH] [EVENT=OPPORTUNITY_TRANSITION] opportunity=jersey_vision_integrator from=EXPLORING to=VALIDATED
[2026-09-19 23:34:19] [test_sess] [RECON] [EVENT=SESSION_END] Overnight mode ended explicitly.
[2026-09-19 23:34:19] [test_sess] [RECON] [EVENT=SESSION_END] Overnight mode ended explicitly.
[2026-09-19 23:34:19] [test_sess] [COMPLETE] [EVENT=CHECKPOINT] current_action=all_opportunities_validated next_action=morning_report_delivered
[2026-09-19 23:34:19] [test_sess] [DISCOVERY] [EVENT=CHECKPOINT] [⚠️ [GUARD_INTERCEPT] Intercepted attempt to set phase='complete' while overnight budget remains (0.0m / 240.0m). Mandating Rediscovery Phase instead of early exit.]
[2026-09-19 23:34:19] [session_night_20260920] [GRAPH] [EVENT=EXTERNAL_RESEARCH] opportunity=sample_op decision=hybrid sources=2
[2026-09-19 23:39:34] [session_night_20260920] [GRAPH] [EVENT=OPPORTUNITY_CREATED] opportunity=feature_dispatch_orchestrator status=PARKED
[2026-09-19 23:39:38] [session_night_20260920] [GRAPH] [EVENT=BRANCH_SWITCHED] tier=main from=camera_motion_ransac to=feature_dispatch_orchestrator
[2026-09-19 23:39:38] [session_night_20260920] [GRAPH] [EVENT=OPPORTUNITY_TRANSITION] opportunity=feature_dispatch_orchestrator from=PARKED to=ACTIVATED
[2026-09-19 23:39:42] [session_night_20260920] [GRAPH] [EVENT=OPPORTUNITY_TRANSITION] opportunity=feature_dispatch_orchestrator from=ACTIVATED to=EXPLORING
[2026-09-19 23:39:45] [session_night_20260920] [EXECUTE] [EVENT=CHECKPOINT] Aligning serverless feature dispatch and task registry
[2026-09-19 23:39:58] [session_night_20260920] [GRAPH] [EVENT=EXTERNAL_RESEARCH] opportunity=feature_dispatch_orchestrator decision=adapted sources=2
[2026-09-19 23:48:03] [session_night_20260920] [GRAPH] [EVENT=BRANCH_DEACTIVATED] tier=main opportunity=feature_dispatch_orchestrator status=VALIDATED
[2026-09-19 23:48:03] [session_night_20260920] [GRAPH] [EVENT=OPPORTUNITY_TRANSITION] opportunity=feature_dispatch_orchestrator from=EXPLORING to=VALIDATED
[2026-09-19 23:49:34] [session_night_20260920] [GRAPH] [EVENT=OPPORTUNITY_CREATED] opportunity=merged_pipeline_stream_overlap status=PARKED
[2026-09-19 23:49:40] [session_night_20260920] [GRAPH] [EVENT=BRANCH_SWITCHED] tier=main from=feature_dispatch_orchestrator to=merged_pipeline_stream_overlap
[2026-09-19 23:49:40] [session_night_20260920] [GRAPH] [EVENT=OPPORTUNITY_TRANSITION] opportunity=merged_pipeline_stream_overlap from=PARKED to=ACTIVATED
[2026-09-19 23:49:46] [session_night_20260920] [GRAPH] [EVENT=OPPORTUNITY_TRANSITION] opportunity=merged_pipeline_stream_overlap from=ACTIVATED to=EXPLORING
[2026-09-19 23:49:52] [session_night_20260920] [EXECUTE] [EVENT=CHECKPOINT] Integrating RobustCameraMovementEstimator into merged pipeline and concurrentizing CUDA streams
[2026-09-19 23:49:58] [session_night_20260920] [GRAPH] [EVENT=EXTERNAL_RESEARCH] opportunity=merged_pipeline_stream_overlap decision=adopted sources=2
[2026-09-19 23:52:15] [session_night_20260920] [GRAPH] [EVENT=BRANCH_DEACTIVATED] tier=main opportunity=merged_pipeline_stream_overlap status=VALIDATED
[2026-09-19 23:52:15] [session_night_20260920] [GRAPH] [EVENT=OPPORTUNITY_TRANSITION] opportunity=merged_pipeline_stream_overlap from=EXPLORING to=VALIDATED
[2026-09-19 23:54:06] [session_night_20260920] [GRAPH] [EVENT=OPPORTUNITY_CREATED] opportunity=pitch_keypoint_cache_gating status=PARKED
[2026-09-19 23:54:09] [session_night_20260920] [GRAPH] [EVENT=BRANCH_SWITCHED] tier=main from=merged_pipeline_stream_overlap to=pitch_keypoint_cache_gating
[2026-09-19 23:54:09] [session_night_20260920] [GRAPH] [EVENT=OPPORTUNITY_TRANSITION] opportunity=pitch_keypoint_cache_gating from=PARKED to=ACTIVATED
[2026-09-19 23:54:12] [session_night_20260920] [GRAPH] [EVENT=OPPORTUNITY_TRANSITION] opportunity=pitch_keypoint_cache_gating from=ACTIVATED to=EXPLORING
[2026-09-19 23:54:16] [session_night_20260920] [EXECUTE] [EVENT=CHECKPOINT] Investigating homography and keypoint gating during static camera shots
[2026-09-19 23:54:30] [session_night_20260920] [GRAPH] [EVENT=EXTERNAL_RESEARCH] opportunity=pitch_keypoint_cache_gating decision=hybrid sources=2
[2026-09-20 00:00:17] [session_night_20260920] [GRAPH] [EVENT=BRANCH_DEACTIVATED] tier=main opportunity=pitch_keypoint_cache_gating status=VALIDATED
[2026-09-20 00:00:17] [session_night_20260920] [GRAPH] [EVENT=OPPORTUNITY_TRANSITION] opportunity=pitch_keypoint_cache_gating from=EXPLORING to=VALIDATED
[2026-09-20 00:00:24] [session_night_20260920] [EXECUTE] [EVENT=CHECKPOINT] Validated pitch_keypoint_cache_gating with 13.8x speedup and 13/13 green tests
[2026-09-20 00:02:42] [session_night_20260920] [GRAPH] [EVENT=OPPORTUNITY_CREATED] opportunity=bidirectional_defensive_line_analyzer status=PARKED
[2026-09-20 00:02:50] [session_night_20260920] [GRAPH] [EVENT=BRANCH_SWITCHED] tier=main from=pitch_keypoint_cache_gating to=bidirectional_defensive_line_analyzer
[2026-09-20 00:02:50] [session_night_20260920] [GRAPH] [EVENT=OPPORTUNITY_TRANSITION] opportunity=bidirectional_defensive_line_analyzer from=PARKED to=ACTIVATED
[2026-09-20 00:02:57] [session_night_20260920] [GRAPH] [EVENT=OPPORTUNITY_TRANSITION] opportunity=bidirectional_defensive_line_analyzer from=ACTIVATED to=EXPLORING
[2026-09-20 00:03:07] [session_night_20260920] [EXECUTE] [EVENT=CHECKPOINT] Investigating tactical defensive line models and directional kinematics
[2026-09-20 00:03:30] [session_night_20260920] [GRAPH] [EVENT=EXTERNAL_RESEARCH] opportunity=bidirectional_defensive_line_analyzer decision=adapted sources=2
[2026-09-20 00:06:17] [session_night_20260920] [GRAPH] [EVENT=BRANCH_DEACTIVATED] tier=main opportunity=bidirectional_defensive_line_analyzer status=VALIDATED
[2026-09-20 00:06:17] [session_night_20260920] [GRAPH] [EVENT=OPPORTUNITY_TRANSITION] opportunity=bidirectional_defensive_line_analyzer from=EXPLORING to=VALIDATED
[2026-09-20 00:06:23] [session_night_20260920] [EXECUTE] [EVENT=CHECKPOINT] Validated bidirectional_defensive_line_analyzer with 1.97M FPS and 7/7 green tests
[2026-09-20 00:06:52] [session_night_20260920] [GRAPH] [EVENT=OPPORTUNITY_CREATED] opportunity=sprint_burst_debouncer status=PARKED
[2026-09-20 00:06:58] [session_night_20260920] [GRAPH] [EVENT=BRANCH_SWITCHED] tier=main from=bidirectional_defensive_line_analyzer to=sprint_burst_debouncer
[2026-09-20 00:06:58] [session_night_20260920] [GRAPH] [EVENT=OPPORTUNITY_TRANSITION] opportunity=sprint_burst_debouncer from=PARKED to=ACTIVATED
[2026-09-20 00:07:05] [session_night_20260920] [GRAPH] [EVENT=OPPORTUNITY_TRANSITION] opportunity=sprint_burst_debouncer from=ACTIVATED to=EXPLORING
[2026-09-20 00:07:11] [session_night_20260920] [EXECUTE] [EVENT=CHECKPOINT] Investigating athletic sprint models and hysteresis debouncing
[2026-09-20 00:07:31] [session_night_20260920] [GRAPH] [EVENT=EXTERNAL_RESEARCH] opportunity=sprint_burst_debouncer decision=adapted sources=2
[2026-09-20 00:09:32] [session_night_20260920] [GRAPH] [EVENT=BRANCH_DEACTIVATED] tier=main opportunity=sprint_burst_debouncer status=VALIDATED
[2026-09-20 00:09:32] [session_night_20260920] [GRAPH] [EVENT=OPPORTUNITY_TRANSITION] opportunity=sprint_burst_debouncer from=EXPLORING to=VALIDATED
[2026-09-20 00:09:39] [session_night_20260920] [EXECUTE] [EVENT=CHECKPOINT] Validated sprint_burst_debouncer with 1.37M FPS and 8/8 green tests
[2026-09-20 00:10:13] [session_night_20260920] [GRAPH] [EVENT=OPPORTUNITY_CREATED] opportunity=speed_kinematics_telemetry_breakdown status=PARKED
[2026-09-20 00:10:19] [session_night_20260920] [GRAPH] [EVENT=BRANCH_SWITCHED] tier=main from=sprint_burst_debouncer to=speed_kinematics_telemetry_breakdown
[2026-09-20 00:10:19] [session_night_20260920] [GRAPH] [EVENT=OPPORTUNITY_TRANSITION] opportunity=speed_kinematics_telemetry_breakdown from=PARKED to=ACTIVATED
[2026-09-20 00:10:25] [session_night_20260920] [GRAPH] [EVENT=OPPORTUNITY_TRANSITION] opportunity=speed_kinematics_telemetry_breakdown from=ACTIVATED to=EXPLORING
[2026-09-20 00:10:32] [session_night_20260920] [EXECUTE] [EVENT=CHECKPOINT] Investigating FIFA 5-zone speed distributions and timeline downsampling
[2026-09-20 00:10:53] [session_night_20260920] [GRAPH] [EVENT=EXTERNAL_RESEARCH] opportunity=speed_kinematics_telemetry_breakdown decision=adopted sources=2
[2026-09-20 00:13:58] [session_night_20260920] [GRAPH] [EVENT=BRANCH_DEACTIVATED] tier=main opportunity=speed_kinematics_telemetry_breakdown status=VALIDATED
[2026-09-20 00:13:58] [session_night_20260920] [GRAPH] [EVENT=OPPORTUNITY_TRANSITION] opportunity=speed_kinematics_telemetry_breakdown from=EXPLORING to=VALIDATED
[2026-09-20 00:14:20] [session_night_20260920] [REDISCOVERY] [EVENT=CHECKPOINT] Completed speed_kinematics_telemetry_breakdown validation, initiating mandatory rediscovery
[2026-09-20 00:17:07] [session_night_20260920] [GRAPH] [EVENT=OPPORTUNITY_CREATED] opportunity=minimap_tactical_replay_generator status=PARKED
[2026-09-20 00:17:13] [session_night_20260920] [GRAPH] [EVENT=OPPORTUNITY_CREATED] opportunity=ai_coach_kinematics_telemetry_sync status=PARKED
[2026-09-20 00:17:17] [session_night_20260920] [GRAPH] [EVENT=BRANCH_SWITCHED] tier=main from=speed_kinematics_telemetry_breakdown to=minimap_tactical_replay_generator
[2026-09-20 00:17:17] [session_night_20260920] [GRAPH] [EVENT=OPPORTUNITY_TRANSITION] opportunity=minimap_tactical_replay_generator from=PARKED to=ACTIVATED
[2026-09-20 00:17:22] [session_night_20260920] [GRAPH] [EVENT=OPPORTUNITY_TRANSITION] opportunity=minimap_tactical_replay_generator from=ACTIVATED to=EXPLORING
[2026-09-20 00:17:34] [session_night_20260920] [GRAPH] [EVENT=EXTERNAL_RESEARCH] opportunity=minimap_tactical_replay_generator decision=hybrid sources=2
[2026-09-20 00:17:38] [session_night_20260920] [EXECUTE] [EVENT=CHECKPOINT] Implementing constant O(1) streaming minimap replay generator
[2026-09-20 00:22:00] [session_night_20260920] [GRAPH] [EVENT=BRANCH_DEACTIVATED] tier=main opportunity=minimap_tactical_replay_generator status=VALIDATED
[2026-09-20 00:22:00] [session_night_20260920] [GRAPH] [EVENT=OPPORTUNITY_TRANSITION] opportunity=minimap_tactical_replay_generator from=EXPLORING to=VALIDATED
[2026-09-20 00:22:17] [session_night_20260920] [GRAPH] [EVENT=BRANCH_SWITCHED] tier=main from=minimap_tactical_replay_generator to=ai_coach_kinematics_telemetry_sync
[2026-09-20 00:22:17] [session_night_20260920] [GRAPH] [EVENT=OPPORTUNITY_TRANSITION] opportunity=ai_coach_kinematics_telemetry_sync from=PARKED to=ACTIVATED
[2026-09-20 00:22:22] [session_night_20260920] [GRAPH] [EVENT=OPPORTUNITY_TRANSITION] opportunity=ai_coach_kinematics_telemetry_sync from=ACTIVATED to=EXPLORING
[2026-09-20 00:22:30] [session_night_20260920] [GRAPH] [EVENT=EXTERNAL_RESEARCH] opportunity=ai_coach_kinematics_telemetry_sync decision=hybrid sources=2
[2026-09-20 00:22:35] [session_night_20260920] [EXECUTE] [EVENT=CHECKPOINT] Synchronizing SprintBurstDebouncer, DefensiveLineAnalyzer, and SpeedTelemetryEngine into AI Coach pipeline
[2026-09-20 00:28:13] [session_night_20260920] [GRAPH] [EVENT=BRANCH_DEACTIVATED] tier=main opportunity=ai_coach_kinematics_telemetry_sync status=VALIDATED
[2026-09-20 00:28:13] [session_night_20260920] [GRAPH] [EVENT=OPPORTUNITY_TRANSITION] opportunity=ai_coach_kinematics_telemetry_sync from=EXPLORING to=VALIDATED
[2026-09-20 00:28:32] [session_night_20260920] [REDISCOVERY] [EVENT=CHECKPOINT] Completed ai_coach_kinematics_telemetry_sync, initiating next observation & discovery cycle
[2026-09-20 00:29:48] [session_night_20260920] [GRAPH] [EVENT=OPPORTUNITY_CREATED] opportunity=tactical_xg_shot_event_detector status=PARKED
[2026-09-20 00:29:52] [session_night_20260920] [GRAPH] [EVENT=BRANCH_SWITCHED] tier=main from=ai_coach_kinematics_telemetry_sync to=tactical_xg_shot_event_detector
[2026-09-20 00:29:52] [session_night_20260920] [GRAPH] [EVENT=OPPORTUNITY_TRANSITION] opportunity=tactical_xg_shot_event_detector from=PARKED to=ACTIVATED
[2026-09-20 00:29:56] [session_night_20260920] [GRAPH] [EVENT=OPPORTUNITY_TRANSITION] opportunity=tactical_xg_shot_event_detector from=ACTIVATED to=EXPLORING
[2026-09-20 00:29:59] [session_night_20260920] [EXPLORING] [EVENT=CHECKPOINT] Starting Opportunity 23: tactical_xg_shot_event_detector
[2026-09-20 00:30:08] [session_night_20260920] [GRAPH] [EVENT=EXTERNAL_RESEARCH] opportunity=tactical_xg_shot_event_detector decision=adapted sources=4
[2026-09-20 00:37:31] [session_night_20260920] [GRAPH] [EVENT=BRANCH_DEACTIVATED] tier=main opportunity=tactical_xg_shot_event_detector status=VALIDATED
[2026-09-20 00:37:31] [session_night_20260920] [GRAPH] [EVENT=OPPORTUNITY_TRANSITION] opportunity=tactical_xg_shot_event_detector from=EXPLORING to=VALIDATED
[2026-09-20 00:37:36] [session_night_20260920] [REDISCOVERY] [EVENT=CHECKPOINT] Opportunity 23 (tactical_xg_shot_event_detector) validated. Entering Mandatory Rediscovery Phase.
[2026-09-20 00:38:06] [session_night_20260920] [GRAPH] [EVENT=OPPORTUNITY_CREATED] opportunity=pressing_intensity_ppda_engine status=PARKED
[2026-09-20 00:38:13] [session_night_20260920] [GRAPH] [EVENT=BRANCH_SWITCHED] tier=main from=tactical_xg_shot_event_detector to=pressing_intensity_ppda_engine
[2026-09-20 00:38:13] [session_night_20260920] [GRAPH] [EVENT=OPPORTUNITY_TRANSITION] opportunity=pressing_intensity_ppda_engine from=PARKED to=ACTIVATED
[2026-09-20 00:38:18] [session_night_20260920] [GRAPH] [EVENT=OPPORTUNITY_TRANSITION] opportunity=pressing_intensity_ppda_engine from=ACTIVATED to=EXPLORING
[2026-09-20 00:38:23] [session_night_20260920] [EXPLORING] [EVENT=CHECKPOINT] Starting Opportunity 24: pressing_intensity_ppda_engine
[2026-09-20 00:38:32] [session_night_20260920] [GRAPH] [EVENT=EXTERNAL_RESEARCH] opportunity=pressing_intensity_ppda_engine decision=adapted sources=4
[2026-09-20 00:43:10] [session_night_20260920] [GRAPH] [EVENT=BRANCH_DEACTIVATED] tier=main opportunity=pressing_intensity_ppda_engine status=VALIDATED
[2026-09-20 00:43:10] [session_night_20260920] [GRAPH] [EVENT=OPPORTUNITY_TRANSITION] opportunity=pressing_intensity_ppda_engine from=EXPLORING to=VALIDATED
[2026-09-20 00:43:16] [session_night_20260920] [REDISCOVERY] [EVENT=CHECKPOINT] Opportunity 24 (pressing_intensity_ppda_engine) validated. Entering Mandatory Rediscovery Phase.
[2026-09-20 00:43:43] [session_night_20260920] [GRAPH] [EVENT=OPPORTUNITY_CREATED] opportunity=match_report_bundle_packager status=PARKED
[2026-09-20 00:43:49] [session_night_20260920] [GRAPH] [EVENT=BRANCH_SWITCHED] tier=main from=pressing_intensity_ppda_engine to=match_report_bundle_packager
[2026-09-20 00:43:49] [session_night_20260920] [GRAPH] [EVENT=OPPORTUNITY_TRANSITION] opportunity=match_report_bundle_packager from=PARKED to=ACTIVATED
[2026-09-20 00:43:55] [session_night_20260920] [GRAPH] [EVENT=OPPORTUNITY_TRANSITION] opportunity=match_report_bundle_packager from=ACTIVATED to=EXPLORING
[2026-09-20 00:44:00] [session_night_20260920] [EXPLORING] [EVENT=CHECKPOINT] Starting Opportunity 25: match_report_bundle_packager
[2026-09-20 00:44:08] [session_night_20260920] [GRAPH] [EVENT=EXTERNAL_RESEARCH] opportunity=match_report_bundle_packager decision=adapted sources=3
[2026-09-20 00:46:35] [session_night_20260920] [GRAPH] [EVENT=BRANCH_DEACTIVATED] tier=main opportunity=match_report_bundle_packager status=VALIDATED
[2026-09-20 00:46:35] [session_night_20260920] [GRAPH] [EVENT=OPPORTUNITY_TRANSITION] opportunity=match_report_bundle_packager from=EXPLORING to=VALIDATED
[2026-09-20 00:46:44] [session_night_20260920] [REDISCOVERY] [EVENT=CHECKPOINT] Opportunity 25 (match_report_bundle_packager) validated. Entering Mandatory Rediscovery Phase.
[2026-09-20 00:47:17] [session_night_20260920] [GRAPH] [EVENT=OPPORTUNITY_CREATED] opportunity=turnover_counterpress_transition_spotter status=PARKED
[2026-09-20 00:47:27] [session_night_20260920] [GRAPH] [EVENT=BRANCH_SWITCHED] tier=main from=match_report_bundle_packager to=turnover_counterpress_transition_spotter
[2026-09-20 00:47:27] [session_night_20260920] [GRAPH] [EVENT=OPPORTUNITY_TRANSITION] opportunity=turnover_counterpress_transition_spotter from=PARKED to=ACTIVATED
[2026-09-20 00:47:37] [session_night_20260920] [GRAPH] [EVENT=OPPORTUNITY_TRANSITION] opportunity=turnover_counterpress_transition_spotter from=ACTIVATED to=EXPLORING
[2026-09-20 00:47:47] [session_night_20260920] [EXPLORING] [EVENT=CHECKPOINT] Starting Opportunity 26: turnover_counterpress_transition_spotter
[2026-09-20 00:48:00] [session_night_20260920] [GRAPH] [EVENT=EXTERNAL_RESEARCH] opportunity=turnover_counterpress_transition_spotter decision=adapted sources=4
[2026-09-20 11:15:25] [session_night_20260920] [GRAPH] [EVENT=BRANCH_DEACTIVATED] tier=main opportunity=turnover_counterpress_transition_spotter status=VALIDATED
[2026-09-20 11:15:25] [session_night_20260920] [GRAPH] [EVENT=OPPORTUNITY_TRANSITION] opportunity=turnover_counterpress_transition_spotter from=EXPLORING to=VALIDATED
[2026-09-20 11:15:50] [session_night_20260920] [EXPLORING] [EVENT=SESSION_END] Overnight mode ended explicitly.
[2026-09-20 11:15:54] [session_night_20260920] [COMPLETED] [EVENT=CHECKPOINT] Overnight R&D session successfully completed. 26/26 opportunities validated, 184/184 tests passing.
[2026-09-20 12:14:11] [test_sess] [RECON] [EVENT=SESSION_END] Overnight mode ended explicitly.
[2026-09-20 12:14:11] [test_sess] [RECON] [EVENT=SESSION_END] Overnight mode ended explicitly.
[2026-09-20 12:14:11] [test_sess] [COMPLETE] [EVENT=CHECKPOINT] current_action=all_opportunities_validated next_action=morning_report_delivered
[2026-09-20 12:14:11] [test_sess] [DISCOVERY] [EVENT=CHECKPOINT] [⚠️ [GUARD_INTERCEPT] Intercepted attempt to set phase='complete' while overnight budget remains (0.0m / 240.0m). Mandating Rediscovery Phase instead of early exit.]
[2026-09-20 12:14:11] [session_night_20260920] [GRAPH] [EVENT=EXTERNAL_RESEARCH] opportunity=sample_op decision=hybrid sources=2
[2026-09-20 12:21:45] [test_sess] [RECON] [EVENT=SESSION_END] Overnight mode ended explicitly.
[2026-09-20 12:21:45] [test_sess] [RECON] [EVENT=SESSION_END] Overnight mode ended explicitly.
[2026-09-20 12:21:45] [test_sess] [COMPLETE] [EVENT=CHECKPOINT] current_action=all_opportunities_validated next_action=morning_report_delivered
[2026-09-20 12:21:45] [test_sess] [DISCOVERY] [EVENT=CHECKPOINT] [⚠️ [GUARD_INTERCEPT] Intercepted attempt to set phase='complete' while overnight budget remains (0.0m / 240.0m). Mandating Rediscovery Phase instead of early exit.]
[2026-09-20 12:21:45] [session_night_20260920] [GRAPH] [EVENT=EXTERNAL_RESEARCH] opportunity=sample_op decision=hybrid sources=2
[2026-09-20 12:26:01] [test_finalize_sess] [EXPLORING] [EVENT=SESSION_END] Overnight mode ended explicitly.
[2026-09-20 12:26:01] [test_finalize_sess] [COMPLETED] [EVENT=SESSION_FINALIZED] Session finalized via canonical shutdown sequence.
[2026-09-20 12:26:01] [test_sess] [RECON] [EVENT=SESSION_END] Overnight mode ended explicitly.
[2026-09-20 12:26:01] [test_sess] [RECON] [EVENT=SESSION_END] Overnight mode ended explicitly.
[2026-09-20 12:26:01] [test_sess] [COMPLETE] [EVENT=CHECKPOINT] current_action=all_opportunities_validated next_action=morning_report_delivered
[2026-09-20 12:26:01] [test_sess] [DISCOVERY] [EVENT=CHECKPOINT] [⚠️ [GUARD_INTERCEPT] Intercepted attempt to set phase='complete' while overnight budget remains (0.0m / 240.0m). Mandating Rediscovery Phase instead of early exit.]
[2026-09-20 12:26:01] [session_night_20260920] [GRAPH] [EVENT=EXTERNAL_RESEARCH] opportunity=sample_op decision=hybrid sources=2
[2026-09-20 12:26:01] [test_sess_deadline] [RECON] [EVENT=SESSION_START] session=test_sess_deadline overnight=True budget=180m deadline=2026-09-20T03:26:01.573268+00:00 mainline=none secondary=none
[2026-09-20 12:26:33] [test_finalize_sess] [EXPLORING] [EVENT=SESSION_END] Overnight mode ended explicitly.
[2026-09-20 12:26:34] [test_finalize_sess] [COMPLETED] [EVENT=SESSION_FINALIZED] Session finalized via canonical shutdown sequence.
[2026-09-20 12:26:34] [test_sess] [RECON] [EVENT=SESSION_END] Overnight mode ended explicitly.
[2026-09-20 12:26:34] [test_sess] [RECON] [EVENT=SESSION_END] Overnight mode ended explicitly.
[2026-09-20 12:26:34] [test_sess] [COMPLETE] [EVENT=CHECKPOINT] current_action=all_opportunities_validated next_action=morning_report_delivered
[2026-09-20 12:26:34] [test_sess] [DISCOVERY] [EVENT=CHECKPOINT] [⚠️ [GUARD_INTERCEPT] Intercepted attempt to set phase='complete' while overnight budget remains (0.0m / 240.0m). Mandating Rediscovery Phase instead of early exit.]
[2026-09-20 12:26:34] [session_night_20260920] [GRAPH] [EVENT=EXTERNAL_RESEARCH] opportunity=sample_op decision=hybrid sources=2
[2026-09-20 12:26:34] [test_sess_deadline] [RECON] [EVENT=SESSION_START] session=test_sess_deadline overnight=True budget=180m deadline=2026-09-20T03:26:34.021202+00:00 mainline=none secondary=none
[2026-09-21 01:40:58] [test_finalize_sess] [EXPLORING] [EVENT=SESSION_END] Overnight mode ended explicitly.
[2026-09-21 01:40:58] [test_finalize_sess] [COMPLETED] [EVENT=SESSION_FINALIZED] Session finalized via canonical shutdown sequence.
[2026-09-21 01:40:58] [test_sess] [RECON] [EVENT=SESSION_END] Overnight mode ended explicitly.
[2026-09-21 01:40:58] [test_sess] [RECON] [EVENT=SESSION_END] Overnight mode ended explicitly.
[2026-09-21 01:40:58] [test_sess] [COMPLETE] [EVENT=CHECKPOINT] current_action=all_opportunities_validated next_action=morning_report_delivered
[2026-09-21 01:40:58] [test_sess] [DISCOVERY] [EVENT=CHECKPOINT] [⚠️ [GUARD_INTERCEPT] Intercepted attempt to set phase='complete' while overnight budget remains (0.0m / 240.0m). Mandating Rediscovery Phase instead of early exit.]
[2026-09-21 01:40:58] [session_night_20260920] [GRAPH] [EVENT=EXTERNAL_RESEARCH] opportunity=sample_op decision=hybrid sources=2
[2026-09-21 01:40:58] [test_sess_deadline] [RECON] [EVENT=SESSION_START] session=test_sess_deadline overnight=True budget=180m deadline=2026-09-20T16:40:58.312097+00:00 mainline=none secondary=none
[2026-09-21 01:44:49] [session_night_20260921] [COMPLETED] [EVENT=SESSION_START] session=session_night_20260921 overnight=True budget=240m deadline=2026-09-20T17:44:49.764061+00:00 mainline=none secondary=none
[2026-09-21 01:44:58] [session_night_20260921] [RECON] [EVENT=CHECKPOINT] Initiated independent session session_night_20260921
[2026-09-21 01:46:04] [session_night_20260921] [RECON] [EVENT=CHECKPOINT] Resolved BUG_PASS_EVENT_TEAM_ATTRIBUTE_DRIFT with full test pass
[2026-09-21 01:46:15] [session_night_20260921] [GRAPH] [EVENT=OPPORTUNITY_CREATED] opportunity=offside_var_freeze_evaluator status=PARKED
[2026-09-21 01:46:20] [session_night_20260921] [GRAPH] [EVENT=OPPORTUNITY_CREATED] opportunity=team_compactness_convex_hull status=PARKED
[2026-09-21 01:46:49] [session_night_20260921] [GRAPH] [EVENT=EXTERNAL_RESEARCH] opportunity=offside_var_freeze_evaluator decision=hybrid sources=3
[2026-09-21 01:46:54] [session_night_20260921] [GRAPH] [EVENT=EXTERNAL_RESEARCH] opportunity=team_compactness_convex_hull decision=adopted sources=3
[2026-09-21 01:47:00] [session_night_20260921] [GRAPH] [EVENT=BRANCH_ACTIVATED] tier=main opportunity=offside_var_freeze_evaluator
[2026-09-21 01:47:00] [session_night_20260921] [EXPLORING] [EVENT=STATE_SYNC] opportunity=offside_var_freeze_evaluator mainline=offside_var_freeze_evaluator secondary=none phase=exploring
[2026-09-21 01:47:00] [session_night_20260921] [GRAPH] [EVENT=OPPORTUNITY_TRANSITION] opportunity=offside_var_freeze_evaluator from=PARKED to=ACTIVATED
[2026-09-21 01:47:04] [session_night_20260921] [EXPLORING] [EVENT=STATE_SYNC] opportunity=offside_var_freeze_evaluator mainline=offside_var_freeze_evaluator secondary=none phase=exploring
[2026-09-21 01:47:04] [session_night_20260921] [GRAPH] [EVENT=OPPORTUNITY_TRANSITION] opportunity=offside_var_freeze_evaluator from=ACTIVATED to=EXPLORING
[2026-09-21 01:47:08] [session_night_20260921] [GRAPH] [EVENT=BRANCH_ACTIVATED] tier=secondary opportunity=team_compactness_convex_hull
[2026-09-21 01:47:08] [session_night_20260921] [EXPLORING] [EVENT=STATE_SYNC] opportunity=team_compactness_convex_hull mainline=offside_var_freeze_evaluator secondary=team_compactness_convex_hull phase=exploring
[2026-09-21 01:47:08] [session_night_20260921] [GRAPH] [EVENT=OPPORTUNITY_TRANSITION] opportunity=team_compactness_convex_hull from=PARKED to=ACTIVATED
[2026-09-21 01:47:13] [session_night_20260921] [EXPLORING] [EVENT=STATE_SYNC] opportunity=team_compactness_convex_hull mainline=offside_var_freeze_evaluator secondary=team_compactness_convex_hull phase=exploring
[2026-09-21 01:47:13] [session_night_20260921] [GRAPH] [EVENT=OPPORTUNITY_TRANSITION] opportunity=team_compactness_convex_hull from=ACTIVATED to=EXPLORING
[2026-09-21 01:49:40] [session_night_20260921] [GRAPH] [EVENT=BRANCH_DEACTIVATED] tier=main opportunity=offside_var_freeze_evaluator status=VALIDATED
[2026-09-21 01:49:40] [session_night_20260921] [EXPLORING] [EVENT=STATE_SYNC] opportunity=offside_var_freeze_evaluator mainline=none secondary=team_compactness_convex_hull phase=exploring
[2026-09-21 01:49:40] [session_night_20260921] [GRAPH] [EVENT=OPPORTUNITY_TRANSITION] opportunity=offside_var_freeze_evaluator from=EXPLORING to=VALIDATED
[2026-09-21 01:52:18] [session_night_20260921] [GRAPH] [EVENT=BRANCH_DEACTIVATED] tier=secondary opportunity=team_compactness_convex_hull status=VALIDATED
[2026-09-21 01:52:18] [session_night_20260921] [REDISCOVERY] [EVENT=STATE_SYNC] opportunity=team_compactness_convex_hull mainline=none secondary=none phase=rediscovery
[2026-09-21 01:52:18] [session_night_20260921] [GRAPH] [EVENT=OPPORTUNITY_TRANSITION] opportunity=team_compactness_convex_hull from=EXPLORING to=VALIDATED
[2026-09-21 01:54:22] [session_night_20260921] [REDISCOVERY] [EVENT=SESSION_END] Overnight mode ended explicitly.
[2026-09-21 01:54:22] [session_night_20260921] [COMPLETED] [EVENT=SESSION_FINALIZED] Session finalized via canonical shutdown sequence.
[2026-09-21 01:54:40] [test_finalize_sess] [EXPLORING] [EVENT=SESSION_END] Overnight mode ended explicitly.
[2026-09-21 01:54:40] [test_finalize_sess] [COMPLETED] [EVENT=SESSION_FINALIZED] Session finalized via canonical shutdown sequence.
[2026-09-21 01:54:40] [test_sess] [RECON] [EVENT=SESSION_END] Overnight mode ended explicitly.
[2026-09-21 01:54:40] [test_sess] [RECON] [EVENT=SESSION_END] Overnight mode ended explicitly.
[2026-09-21 01:54:40] [test_sess] [COMPLETE] [EVENT=CHECKPOINT] current_action=all_opportunities_validated next_action=morning_report_delivered
[2026-09-21 01:54:40] [test_sess] [DISCOVERY] [EVENT=CHECKPOINT] [⚠️ [GUARD_INTERCEPT] Intercepted attempt to set phase='complete' while overnight budget remains (0.0m / 240.0m). Mandating Rediscovery Phase instead of early exit.]
[2026-09-21 01:54:40] [session_night_20260921] [GRAPH] [EVENT=EXTERNAL_RESEARCH] opportunity=sample_op decision=hybrid sources=2
[2026-09-21 01:54:40] [test_sess_deadline] [RECON] [EVENT=SESSION_START] session=test_sess_deadline overnight=True budget=180m deadline=2026-09-20T16:54:40.024231+00:00 mainline=none secondary=none
[2026-09-21 09:56:29] [session_night_20260921_perf] [COMPLETED] [EVENT=SESSION_START] session=session_night_20260921_perf overnight=True budget=240m deadline=2026-09-21T01:56:29.228035+00:00 mainline=none secondary=none
[2026-09-21 09:56:38] [session_night_20260921_perf] [GRAPH] [EVENT=OPPORTUNITY_CREATED] opportunity=e2e_video_detection_profiler status=PARKED
[2026-09-21 09:56:41] [session_night_20260921_perf] [GRAPH] [EVENT=OPPORTUNITY_CREATED] opportunity=detection_tracking_pareto_evaluator status=PARKED
[2026-09-21 09:56:44] [session_night_20260921_perf] [GRAPH] [EVENT=BRANCH_ACTIVATED] tier=main opportunity=e2e_video_detection_profiler
[2026-09-21 09:56:44] [session_night_20260921_perf] [EXPLORING] [EVENT=STATE_SYNC] opportunity=e2e_video_detection_profiler mainline=e2e_video_detection_profiler secondary=none phase=exploring
[2026-09-21 09:56:44] [session_night_20260921_perf] [GRAPH] [EVENT=OPPORTUNITY_TRANSITION] opportunity=e2e_video_detection_profiler from=PARKED to=ACTIVATED
[2026-09-21 09:56:46] [session_night_20260921_perf] [GRAPH] [EVENT=BRANCH_ACTIVATED] tier=secondary opportunity=detection_tracking_pareto_evaluator
[2026-09-21 09:56:46] [session_night_20260921_perf] [EXPLORING] [EVENT=STATE_SYNC] opportunity=detection_tracking_pareto_evaluator mainline=e2e_video_detection_profiler secondary=detection_tracking_pareto_evaluator phase=exploring
[2026-09-21 09:56:46] [session_night_20260921_perf] [GRAPH] [EVENT=OPPORTUNITY_TRANSITION] opportunity=detection_tracking_pareto_evaluator from=PARKED to=ACTIVATED
[2026-09-21 09:56:49] [session_night_20260921_perf] [DISCOVERY] [EVENT=CHECKPOINT] Activated speed & accuracy P0 tasks
[2026-09-21 09:56:53] [session_night_20260921_perf] [GRAPH] [EVENT=EXTERNAL_RESEARCH] opportunity=e2e_video_detection_profiler decision=adapted sources=4
[2026-09-21 09:56:56] [session_night_20260921_perf] [GRAPH] [EVENT=EXTERNAL_RESEARCH] opportunity=detection_tracking_pareto_evaluator decision=adapted sources=4
[2026-09-21 09:56:58] [session_night_20260921_perf] [DISCOVERY] [EVENT=STATE_SYNC] opportunity=e2e_video_detection_profiler mainline=e2e_video_detection_profiler secondary=detection_tracking_pareto_evaluator phase=discovery
[2026-09-21 09:56:58] [session_night_20260921_perf] [GRAPH] [EVENT=OPPORTUNITY_TRANSITION] opportunity=e2e_video_detection_profiler from=ACTIVATED to=EXPLORING
[2026-09-21 09:57:01] [session_night_20260921_perf] [DISCOVERY] [EVENT=STATE_SYNC] opportunity=detection_tracking_pareto_evaluator mainline=e2e_video_detection_profiler secondary=detection_tracking_pareto_evaluator phase=discovery
[2026-09-21 09:57:01] [session_night_20260921_perf] [GRAPH] [EVENT=OPPORTUNITY_TRANSITION] opportunity=detection_tracking_pareto_evaluator from=ACTIVATED to=EXPLORING
[2026-09-21 10:09:05] [session_night_20260921_perf] [GRAPH] [EVENT=BRANCH_DEACTIVATED] tier=main opportunity=e2e_video_detection_profiler status=VALIDATED
[2026-09-21 10:09:05] [session_night_20260921_perf] [DISCOVERY] [EVENT=STATE_SYNC] opportunity=e2e_video_detection_profiler mainline=none secondary=detection_tracking_pareto_evaluator phase=discovery
[2026-09-21 10:09:05] [session_night_20260921_perf] [GRAPH] [EVENT=OPPORTUNITY_TRANSITION] opportunity=e2e_video_detection_profiler from=EXPLORING to=VALIDATED
[2026-09-21 10:09:10] [session_night_20260921_perf] [GRAPH] [EVENT=BRANCH_DEACTIVATED] tier=secondary opportunity=detection_tracking_pareto_evaluator status=VALIDATED
[2026-09-21 10:09:10] [session_night_20260921_perf] [REDISCOVERY] [EVENT=STATE_SYNC] opportunity=detection_tracking_pareto_evaluator mainline=none secondary=none phase=rediscovery
[2026-09-21 10:09:10] [session_night_20260921_perf] [GRAPH] [EVENT=OPPORTUNITY_TRANSITION] opportunity=detection_tracking_pareto_evaluator from=EXPLORING to=VALIDATED
[2026-09-21 10:11:17] [session_night_20260921_perf] [GRAPH] [EVENT=OPPORTUNITY_CREATED] opportunity=bytetrack_adaptive_iou_stride_compensator status=PARKED
[2026-09-21 10:11:22] [session_night_20260921_perf] [GRAPH] [EVENT=OPPORTUNITY_CREATED] opportunity=yolo_tensorrt_fp16_runpod_quantizer status=PARKED
[2026-09-21 10:11:27] [session_night_20260921_perf] [REDISCOVERY] [EVENT=SESSION_END] Overnight mode ended explicitly.
[2026-09-21 10:11:27] [session_night_20260921_perf] [COMPLETED] [EVENT=SESSION_FINALIZED] Session finalized via canonical shutdown sequence.
[2026-09-21 10:29:49] [test_finalize_sess] [EXPLORING] [EVENT=SESSION_END] Overnight mode ended explicitly.
[2026-09-21 10:29:49] [test_finalize_sess] [COMPLETED] [EVENT=SESSION_FINALIZED] Session finalized via canonical shutdown sequence.
[2026-09-21 10:29:49] [test_sess] [RECON] [EVENT=SESSION_END] Overnight mode ended explicitly.
[2026-09-21 10:29:49] [test_sess] [RECON] [EVENT=SESSION_END] Overnight mode ended explicitly.
[2026-09-21 10:29:49] [test_sess] [COMPLETE] [EVENT=CHECKPOINT] current_action=all_opportunities_validated next_action=morning_report_delivered
[2026-09-21 10:29:49] [test_sess] [DISCOVERY] [EVENT=CHECKPOINT] [⚠️ [GUARD_INTERCEPT] Intercepted attempt to set phase='complete' while overnight budget remains (0.0m / 240.0m). Mandating Rediscovery Phase instead of early exit.]
[2026-09-21 10:29:49] [session_night_20260921_perf] [GRAPH] [EVENT=EXTERNAL_RESEARCH] opportunity=sample_op decision=hybrid sources=2
[2026-09-21 10:29:49] [test_sess_deadline] [RECON] [EVENT=SESSION_START] session=test_sess_deadline overnight=True budget=180m deadline=2026-09-21T01:29:49.722477+00:00 mainline=none secondary=none
[2026-09-21 10:31:51] [test_finalize_sess] [EXPLORING] [EVENT=SESSION_END] Overnight mode ended explicitly.
[2026-09-21 10:31:51] [test_finalize_sess] [COMPLETED] [EVENT=SESSION_FINALIZED] Session finalized via canonical shutdown sequence.
[2026-09-21 10:31:51] [test_finalize_sess] [EXPLORING] [EVENT=SESSION_END] Overnight mode ended explicitly.
[2026-09-21 10:31:51] [test_finalize_sess] [COMPLETED] [EVENT=SESSION_FINALIZED] Session finalized via canonical shutdown sequence.
[2026-09-21 10:31:51] [test_sess] [RECON] [EVENT=SESSION_END] Overnight mode ended explicitly.
[2026-09-21 10:31:51] [test_sess] [RECON] [EVENT=SESSION_END] Overnight mode ended explicitly.
[2026-09-21 10:31:51] [test_sess] [COMPLETE] [EVENT=CHECKPOINT] current_action=all_opportunities_validated next_action=morning_report_delivered
[2026-09-21 10:31:51] [test_sess] [DISCOVERY] [EVENT=CHECKPOINT] [⚠️ [GUARD_INTERCEPT] Intercepted attempt to set phase='complete' while overnight budget remains (0.0m / 240.0m). Mandating Rediscovery Phase instead of early exit.]
[2026-09-21 10:31:51] [session_night_20260921_perf] [GRAPH] [EVENT=EXTERNAL_RESEARCH] opportunity=sample_op decision=hybrid sources=2
[2026-09-21 10:31:51] [test_sess_deadline] [RECON] [EVENT=SESSION_START] session=test_sess_deadline overnight=True budget=180m deadline=2026-09-21T01:31:51.319112+00:00 mainline=none secondary=none
[2026-09-21 10:31:56] [session_night_20260921_perf] [COMPLETED] [EVENT=SESSION_END] Overnight mode ended explicitly.
[2026-09-21 10:31:56] [session_night_20260921_perf] [COMPLETED] [EVENT=SESSION_FINALIZED] Session finalized via canonical shutdown sequence.
[2026-09-21 10:31:56] [session_night_20260921_perf] [COMPLETED] [EVENT=SESSION_END] Overnight mode ended explicitly.
[2026-09-21 10:31:56] [session_night_20260921_perf] [COMPLETED] [EVENT=SESSION_FINALIZED] Session finalized via canonical shutdown sequence.
[2026-09-21 10:32:38] [test_finalize_sess] [EXPLORING] [EVENT=SESSION_END] Overnight mode ended explicitly.
[2026-09-21 10:32:38] [test_finalize_sess] [COMPLETED] [EVENT=SESSION_FINALIZED] Session finalized via canonical shutdown sequence.
[2026-09-21 10:32:38] [test_finalize_sess] [EXPLORING] [EVENT=SESSION_END] Overnight mode ended explicitly.
[2026-09-21 10:32:38] [test_finalize_sess] [COMPLETED] [EVENT=SESSION_FINALIZED] Session finalized via canonical shutdown sequence.
[2026-09-21 10:32:38] [test_sess] [RECON] [EVENT=SESSION_END] Overnight mode ended explicitly.
[2026-09-21 10:32:38] [test_sess] [RECON] [EVENT=SESSION_END] Overnight mode ended explicitly.
[2026-09-21 10:32:38] [test_sess] [COMPLETE] [EVENT=CHECKPOINT] current_action=all_opportunities_validated next_action=morning_report_delivered
[2026-09-21 10:32:38] [test_sess] [DISCOVERY] [EVENT=CHECKPOINT] [⚠️ [GUARD_INTERCEPT] Intercepted attempt to set phase='complete' while overnight budget remains (0.0m / 240.0m). Mandating Rediscovery Phase instead of early exit.]
[2026-09-21 10:32:38] [session_night_20260921_perf] [GRAPH] [EVENT=EXTERNAL_RESEARCH] opportunity=sample_op decision=hybrid sources=2
[2026-09-21 10:32:38] [test_sess_deadline] [RECON] [EVENT=SESSION_START] session=test_sess_deadline overnight=True budget=180m deadline=2026-09-21T01:32:38.996232+00:00 mainline=none secondary=none
[2026-09-21 10:32:54] [test_finalize_sess] [EXPLORING] [EVENT=SESSION_END] Overnight mode ended explicitly.
[2026-09-21 10:32:54] [test_finalize_sess] [COMPLETED] [EVENT=SESSION_FINALIZED] Session finalized via canonical shutdown sequence.
[2026-09-21 10:32:54] [test_finalize_sess] [EXPLORING] [EVENT=SESSION_END] Overnight mode ended explicitly.
[2026-09-21 10:32:54] [test_finalize_sess] [COMPLETED] [EVENT=SESSION_FINALIZED] Session finalized via canonical shutdown sequence.
[2026-09-21 10:32:54] [test_sess] [RECON] [EVENT=SESSION_END] Overnight mode ended explicitly.
[2026-09-21 10:32:54] [test_sess] [RECON] [EVENT=SESSION_END] Overnight mode ended explicitly.
[2026-09-21 10:32:54] [test_sess] [COMPLETE] [EVENT=CHECKPOINT] current_action=all_opportunities_validated next_action=morning_report_delivered
[2026-09-21 10:32:54] [test_sess] [DISCOVERY] [EVENT=CHECKPOINT] [⚠️ [GUARD_INTERCEPT] Intercepted attempt to set phase='complete' while overnight budget remains (0.0m / 240.0m). Mandating Rediscovery Phase instead of early exit.]
[2026-09-21 10:32:54] [session_night_20260921_perf] [GRAPH] [EVENT=EXTERNAL_RESEARCH] opportunity=sample_op decision=hybrid sources=2
[2026-09-21 10:32:54] [test_sess_deadline] [RECON] [EVENT=SESSION_START] session=test_sess_deadline overnight=True budget=180m deadline=2026-09-21T01:32:54.760152+00:00 mainline=none secondary=none
[2026-09-21 10:57:13] [test_finalize_sess] [EXPLORING] [EVENT=SESSION_END] Overnight mode ended explicitly.
[2026-09-21 10:57:13] [test_finalize_sess] [COMPLETED] [EVENT=SESSION_FINALIZED] Session finalized via canonical shutdown sequence.
[2026-09-21 10:57:13] [test_sess] [RECON] [EVENT=SESSION_END] Overnight mode ended explicitly.
[2026-09-21 10:57:13] [test_sess] [COMPLETE] [EVENT=CHECKPOINT] current_action=all_opportunities_validated next_action=morning_report_delivered
[2026-09-21 10:57:13] [test_sess] [DISCOVERY] [EVENT=CHECKPOINT] [⚠️ [GUARD_INTERCEPT] Intercepted attempt to set phase='complete' while overnight budget remains (0.0m / 240.0m). Mandating Rediscovery Phase instead of early exit.]
[2026-09-21 10:57:13] [session_night_20260921_perf] [GRAPH] [EVENT=EXTERNAL_RESEARCH] opportunity=sample_op decision=hybrid sources=2
[2026-09-21 10:57:13] [test_sess_deadline] [RECON] [EVENT=SESSION_START] session=test_sess_deadline overnight=True budget=180m deadline=2026-09-21T01:57:13.907773+00:00 mainline=none secondary=none
[2026-09-21 10:57:52] [test_finalize_sess] [EXPLORING] [EVENT=SESSION_END] Overnight mode ended explicitly.
[2026-09-21 10:57:52] [test_finalize_sess] [COMPLETED] [EVENT=SESSION_FINALIZED] Session finalized via canonical shutdown sequence.
[2026-09-21 10:57:52] [test_sess] [RECON] [EVENT=SESSION_END] Overnight mode ended explicitly.
[2026-09-21 10:57:52] [test_sess] [COMPLETE] [EVENT=CHECKPOINT] current_action=all_opportunities_validated next_action=morning_report_delivered
[2026-09-21 10:57:52] [test_sess] [DISCOVERY] [EVENT=CHECKPOINT] [⚠️ [GUARD_INTERCEPT] Intercepted attempt to set phase='complete' while overnight budget remains (0.0m / 240.0m). Mandating Rediscovery Phase instead of early exit.]
[2026-09-21 10:57:52] [session_night_20260921_perf] [GRAPH] [EVENT=EXTERNAL_RESEARCH] opportunity=sample_op decision=hybrid sources=2
[2026-09-21 10:57:52] [test_sess_deadline] [RECON] [EVENT=SESSION_START] session=test_sess_deadline overnight=True budget=180m deadline=2026-09-21T01:57:52.491145+00:00 mainline=none secondary=none
[2026-09-21 12:39:15] [session_night_20260921_accel] [COMPLETED] [EVENT=SESSION_START] session=session_night_20260921_accel overnight=True budget=240m deadline=2026-09-21T04:39:15.015399+00:00 mainline=none secondary=none
[2026-09-21 12:39:27] [session_night_20260921_accel] [GRAPH] [EVENT=BRANCH_ACTIVATED] tier=main opportunity=yolo_tensorrt_fp16_runpod_quantizer
[2026-09-21 12:39:27] [session_night_20260921_accel] [EXPLORING] [EVENT=STATE_SYNC] opportunity=yolo_tensorrt_fp16_runpod_quantizer mainline=yolo_tensorrt_fp16_runpod_quantizer secondary=none phase=exploring
[2026-09-21 12:39:27] [session_night_20260921_accel] [GRAPH] [EVENT=OPPORTUNITY_TRANSITION] opportunity=yolo_tensorrt_fp16_runpod_quantizer from=PARKED to=ACTIVATED
[2026-09-21 12:39:32] [session_night_20260921_accel] [GRAPH] [EVENT=BRANCH_ACTIVATED] tier=secondary opportunity=bytetrack_adaptive_iou_stride_compensator
[2026-09-21 12:39:32] [session_night_20260921_accel] [EXPLORING] [EVENT=STATE_SYNC] opportunity=bytetrack_adaptive_iou_stride_compensator mainline=yolo_tensorrt_fp16_runpod_quantizer secondary=bytetrack_adaptive_iou_stride_compensator phase=exploring
[2026-09-21 12:39:32] [session_night_20260921_accel] [GRAPH] [EVENT=OPPORTUNITY_TRANSITION] opportunity=bytetrack_adaptive_iou_stride_compensator from=PARKED to=ACTIVATED
[2026-09-21 12:39:36] [session_night_20260921_accel] [EXPLORING] [EVENT=CHECKPOINT] Activated P0 mainline and secondary opportunities for inference speed and tracking accuracy
[2026-09-21 12:56:03] [session_night_20260921_accel] [EXPLORING] [EVENT=STATE_SYNC] opportunity=yolo_tensorrt_fp16_runpod_quantizer mainline=yolo_tensorrt_fp16_runpod_quantizer secondary=bytetrack_adaptive_iou_stride_compensator phase=exploring
[2026-09-21 12:56:03] [session_night_20260921_accel] [GRAPH] [EVENT=OPPORTUNITY_TRANSITION] opportunity=yolo_tensorrt_fp16_runpod_quantizer from=ACTIVATED to=EXPLORING
[2026-09-21 12:56:09] [session_night_20260921_accel] [EXPLORING] [EVENT=STATE_SYNC] opportunity=bytetrack_adaptive_iou_stride_compensator mainline=yolo_tensorrt_fp16_runpod_quantizer secondary=bytetrack_adaptive_iou_stride_compensator phase=exploring
[2026-09-21 12:56:09] [session_night_20260921_accel] [GRAPH] [EVENT=OPPORTUNITY_TRANSITION] opportunity=bytetrack_adaptive_iou_stride_compensator from=ACTIVATED to=EXPLORING
[2026-09-21 12:56:14] [session_night_20260921_accel] [GRAPH] [EVENT=BRANCH_DEACTIVATED] tier=main opportunity=yolo_tensorrt_fp16_runpod_quantizer status=VALIDATED
[2026-09-21 12:56:14] [session_night_20260921_accel] [EXPLORING] [EVENT=STATE_SYNC] opportunity=yolo_tensorrt_fp16_runpod_quantizer mainline=none secondary=bytetrack_adaptive_iou_stride_compensator phase=exploring
[2026-09-21 12:56:14] [session_night_20260921_accel] [GRAPH] [EVENT=OPPORTUNITY_TRANSITION] opportunity=yolo_tensorrt_fp16_runpod_quantizer from=EXPLORING to=VALIDATED
[2026-09-21 12:56:19] [session_night_20260921_accel] [GRAPH] [EVENT=BRANCH_DEACTIVATED] tier=secondary opportunity=bytetrack_adaptive_iou_stride_compensator status=VALIDATED
[2026-09-21 12:56:19] [session_night_20260921_accel] [REDISCOVERY] [EVENT=STATE_SYNC] opportunity=bytetrack_adaptive_iou_stride_compensator mainline=none secondary=none phase=rediscovery
[2026-09-21 12:56:19] [session_night_20260921_accel] [GRAPH] [EVENT=OPPORTUNITY_TRANSITION] opportunity=bytetrack_adaptive_iou_stride_compensator from=EXPLORING to=VALIDATED
[2026-09-21 12:56:43] [session_night_20260921_accel] [GRAPH] [EVENT=OPPORTUNITY_CREATED] opportunity=pitch_roi_crop_detector status=PARKED
[2026-09-21 12:56:48] [session_night_20260921_accel] [GRAPH] [EVENT=OPPORTUNITY_CREATED] opportunity=ball_trajectory_physics_interpolator status=PARKED
[2026-09-21 12:56:56] [session_night_20260921_accel] [GRAPH] [EVENT=BRANCH_SWITCHED] tier=main from=yolo_tensorrt_fp16_runpod_quantizer to=pitch_roi_crop_detector
[2026-09-21 12:56:56] [session_night_20260921_accel] [EXPLORING] [EVENT=STATE_SYNC] opportunity=pitch_roi_crop_detector mainline=pitch_roi_crop_detector secondary=none phase=exploring
[2026-09-21 12:56:56] [session_night_20260921_accel] [GRAPH] [EVENT=OPPORTUNITY_TRANSITION] opportunity=pitch_roi_crop_detector from=PARKED to=ACTIVATED
[2026-09-21 12:57:01] [session_night_20260921_accel] [GRAPH] [EVENT=BRANCH_SWITCHED] tier=secondary from=bytetrack_adaptive_iou_stride_compensator to=ball_trajectory_physics_interpolator
[2026-09-21 12:57:01] [session_night_20260921_accel] [EXPLORING] [EVENT=STATE_SYNC] opportunity=ball_trajectory_physics_interpolator mainline=pitch_roi_crop_detector secondary=ball_trajectory_physics_interpolator phase=exploring
[2026-09-21 12:57:01] [session_night_20260921_accel] [GRAPH] [EVENT=OPPORTUNITY_TRANSITION] opportunity=ball_trajectory_physics_interpolator from=PARKED to=ACTIVATED
[2026-09-21 12:57:09] [session_night_20260921_accel] [EXPLORING] [EVENT=CHECKPOINT] Activated Rediscovered P0 opportunities for ROI inference speedup and ball trajectory bridging
[2026-09-21 12:57:16] [session_night_20260921_accel] [GRAPH] [EVENT=EXTERNAL_RESEARCH] opportunity=pitch_roi_crop_detector decision=adapted sources=2
[2026-09-21 12:57:22] [session_night_20260921_accel] [GRAPH] [EVENT=EXTERNAL_RESEARCH] opportunity=ball_trajectory_physics_interpolator decision=adapted sources=2
[2026-09-21 12:57:29] [session_night_20260921_accel] [EXPLORING] [EVENT=STATE_SYNC] opportunity=pitch_roi_crop_detector mainline=pitch_roi_crop_detector secondary=ball_trajectory_physics_interpolator phase=exploring
[2026-09-21 12:57:29] [session_night_20260921_accel] [GRAPH] [EVENT=OPPORTUNITY_TRANSITION] opportunity=pitch_roi_crop_detector from=ACTIVATED to=EXPLORING
[2026-09-21 12:57:35] [session_night_20260921_accel] [EXPLORING] [EVENT=STATE_SYNC] opportunity=ball_trajectory_physics_interpolator mainline=pitch_roi_crop_detector secondary=ball_trajectory_physics_interpolator phase=exploring
[2026-09-21 12:57:35] [session_night_20260921_accel] [GRAPH] [EVENT=OPPORTUNITY_TRANSITION] opportunity=ball_trajectory_physics_interpolator from=ACTIVATED to=EXPLORING
[2026-09-21 12:59:11] [session_night_20260921_accel] [GRAPH] [EVENT=BRANCH_DEACTIVATED] tier=main opportunity=pitch_roi_crop_detector status=VALIDATED
[2026-09-21 12:59:11] [session_night_20260921_accel] [EXPLORING] [EVENT=STATE_SYNC] opportunity=pitch_roi_crop_detector mainline=none secondary=ball_trajectory_physics_interpolator phase=exploring
[2026-09-21 12:59:11] [session_night_20260921_accel] [GRAPH] [EVENT=OPPORTUNITY_TRANSITION] opportunity=pitch_roi_crop_detector from=EXPLORING to=VALIDATED
[2026-09-21 12:59:19] [session_night_20260921_accel] [GRAPH] [EVENT=BRANCH_DEACTIVATED] tier=secondary opportunity=ball_trajectory_physics_interpolator status=VALIDATED
[2026-09-21 12:59:19] [session_night_20260921_accel] [REDISCOVERY] [EVENT=STATE_SYNC] opportunity=ball_trajectory_physics_interpolator mainline=none secondary=none phase=rediscovery
[2026-09-21 12:59:19] [session_night_20260921_accel] [GRAPH] [EVENT=OPPORTUNITY_TRANSITION] opportunity=ball_trajectory_physics_interpolator from=EXPLORING to=VALIDATED
[2026-09-21 12:59:55] [session_night_20260921_accel] [GRAPH] [EVENT=EXTERNAL_RESEARCH] opportunity=sample_op decision=hybrid sources=2
[2026-09-21 12:59:55] [test_sess] [RECON] [EVENT=SESSION_END] Overnight mode ended explicitly.
[2026-09-21 12:59:55] [test_sess] [COMPLETE] [EVENT=CHECKPOINT] current_action=all_opportunities_validated next_action=morning_report_delivered
[2026-09-21 12:59:55] [test_sess] [DISCOVERY] [EVENT=CHECKPOINT] [⚠️ [GUARD_INTERCEPT] Intercepted attempt to set phase='complete' while overnight budget remains (0.0m / 240.0m). Mandating Rediscovery Phase instead of early exit.]
[2026-09-21 12:59:55] [test_sess_deadline] [RECON] [EVENT=SESSION_START] session=test_sess_deadline overnight=True budget=180m deadline=2026-09-21T03:59:55.040399+00:00 mainline=none secondary=none
[2026-09-21 12:59:55] [test_finalize_sess] [EXPLORING] [EVENT=SESSION_END] Overnight mode ended explicitly.
[2026-09-21 12:59:55] [test_finalize_sess] [COMPLETED] [EVENT=SESSION_FINALIZED] Session finalized via canonical shutdown sequence.
[2026-09-21 14:14:30] [test_finalize_sess] [EXPLORING] [EVENT=SESSION_END] Overnight mode ended explicitly.
[2026-09-21 14:14:30] [test_finalize_sess] [COMPLETED] [EVENT=SESSION_FINALIZED] Session finalized via canonical shutdown sequence.
[2026-09-21 14:14:30] [test_sess] [RECON] [EVENT=SESSION_END] Overnight mode ended explicitly.
[2026-09-21 14:14:30] [test_sess] [COMPLETE] [EVENT=CHECKPOINT] current_action=all_opportunities_validated next_action=morning_report_delivered
[2026-09-21 14:14:30] [test_sess] [DISCOVERY] [EVENT=CHECKPOINT] [⚠️ [GUARD_INTERCEPT] Intercepted attempt to set phase='complete' while overnight budget remains (0.0m / 240.0m). Mandating Rediscovery Phase instead of early exit.]
[2026-09-21 14:14:30] [session_night_20260921_accel] [GRAPH] [EVENT=EXTERNAL_RESEARCH] opportunity=sample_op decision=hybrid sources=2
[2026-09-21 14:14:30] [test_sess_deadline] [RECON] [EVENT=SESSION_START] session=test_sess_deadline overnight=True budget=180m deadline=2026-09-21T05:14:30.369038+00:00 mainline=none secondary=none
[2026-09-21 14:29:46] [session_night_20260921_accel] [REDISCOVERY] [EVENT=STATE_SYNC] opportunity=offside_var_freeze_evaluator mainline=none secondary=none phase=rediscovery
[2026-09-21 14:29:46] [session_night_20260921_accel] [GRAPH] [EVENT=OPPORTUNITY_TRANSITION] opportunity=offside_var_freeze_evaluator from=VALIDATED_SANDBOX to=PRODUCTION_CANDIDATE
[2026-09-21 14:29:52] [session_night_20260921_accel] [REDISCOVERY] [EVENT=STATE_SYNC] opportunity=team_compactness_convex_hull mainline=none secondary=none phase=rediscovery
[2026-09-21 14:29:52] [session_night_20260921_accel] [GRAPH] [EVENT=OPPORTUNITY_TRANSITION] opportunity=team_compactness_convex_hull from=VALIDATED_SANDBOX to=PRODUCTION_CANDIDATE
[2026-09-21 14:31:17] [test_finalize_sess] [EXPLORING] [EVENT=SESSION_END] Overnight mode ended explicitly.
[2026-09-21 14:31:17] [test_finalize_sess] [COMPLETED] [EVENT=SESSION_FINALIZED] Session finalized via canonical shutdown sequence.
[2026-09-21 14:31:17] [test_sess] [RECON] [EVENT=SESSION_END] Overnight mode ended explicitly.
[2026-09-21 14:31:17] [test_sess] [COMPLETE] [EVENT=CHECKPOINT] current_action=all_opportunities_validated next_action=morning_report_delivered
[2026-09-21 14:31:17] [test_sess] [DISCOVERY] [EVENT=CHECKPOINT] [⚠️ [GUARD_INTERCEPT] Intercepted attempt to set phase='complete' while overnight budget remains (0.0m / 240.0m). Mandating Rediscovery Phase instead of early exit.]
[2026-09-21 14:31:17] [session_night_20260921_accel] [GRAPH] [EVENT=EXTERNAL_RESEARCH] opportunity=sample_op decision=hybrid sources=2
[2026-09-21 14:31:17] [test_sess_deadline] [RECON] [EVENT=SESSION_START] session=test_sess_deadline overnight=True budget=180m deadline=2026-09-21T05:31:17.378288+00:00 mainline=none secondary=none
[2026-09-21 22:49:31] [test_finalize_sess] [EXPLORING] [EVENT=SESSION_END] Overnight mode ended explicitly.
[2026-09-21 22:49:31] [test_finalize_sess] [COMPLETED] [EVENT=SESSION_FINALIZED] Session finalized via canonical shutdown sequence.
[2026-09-21 22:49:31] [test_sess] [RECON] [EVENT=SESSION_END] Overnight mode ended explicitly.
[2026-09-21 22:49:31] [test_sess] [COMPLETE] [EVENT=CHECKPOINT] current_action=all_opportunities_validated next_action=morning_report_delivered
[2026-09-21 22:49:31] [test_sess] [DISCOVERY] [EVENT=CHECKPOINT] [⚠️ [GUARD_INTERCEPT] Intercepted attempt to set phase='complete' while overnight budget remains (0.0m / 240.0m). Mandating Rediscovery Phase instead of early exit.]
[2026-09-21 22:49:31] [session_night_20260921_accel] [GRAPH] [EVENT=EXTERNAL_RESEARCH] opportunity=sample_op decision=hybrid sources=2
[2026-09-21 22:49:31] [test_sess_deadline] [RECON] [EVENT=SESSION_START] session=test_sess_deadline overnight=True budget=180m deadline=2026-09-21T13:49:31.899773+00:00 mainline=none secondary=none
[2026-09-21 22:52:40] [session_night_20260921_accel] [REDISCOVERY] [EVENT=STATE_SYNC] opportunity=e2e_video_detection_profiler mainline=none secondary=none phase=rediscovery
[2026-09-21 22:52:40] [session_night_20260921_accel] [GRAPH] [EVENT=OPPORTUNITY_TRANSITION] opportunity=e2e_video_detection_profiler from=VALIDATED_SANDBOX to=PRODUCTION_CANDIDATE
[2026-09-21 22:52:44] [session_night_20260921_accel] [REDISCOVERY] [EVENT=STATE_SYNC] opportunity=detection_tracking_pareto_evaluator mainline=none secondary=none phase=rediscovery
[2026-09-21 22:52:44] [session_night_20260921_accel] [GRAPH] [EVENT=OPPORTUNITY_TRANSITION] opportunity=detection_tracking_pareto_evaluator from=VALIDATED_SANDBOX to=PRODUCTION_CANDIDATE
[2026-09-21 22:53:04] [session_night_20260921_accel] [GRAPH] [EVENT=OPPORTUNITY_CREATED] opportunity=samurai_bounded_memory_pool status=PARKED
[2026-09-21 22:53:08] [session_night_20260921_accel] [GRAPH] [EVENT=OPPORTUNITY_CREATED] opportunity=supabase_rpc_merge_session_extra_fix status=PARKED
[2026-09-21 22:53:21] [session_night_20260921_accel] [REDISCOVERY] [EVENT=CHECKPOINT] P0 Pareto benchmark verified on 150 frames real 1080p video. 1280p stride=3 achieves 78.3 FPS (RTF 0.32). Candidates promoted under Production Gate. Discovered samurai_bounded_memory_pool and supabase_rpc_merge_session_extra_fix from RunPod logs.
[2026-09-21 22:54:41] [test_finalize_sess] [EXPLORING] [EVENT=SESSION_END] Overnight mode ended explicitly.
[2026-09-21 22:54:41] [test_finalize_sess] [COMPLETED] [EVENT=SESSION_FINALIZED] Session finalized via canonical shutdown sequence.
[2026-09-21 22:54:41] [test_sess] [RECON] [EVENT=SESSION_END] Overnight mode ended explicitly.
[2026-09-21 22:54:41] [test_sess] [COMPLETE] [EVENT=CHECKPOINT] current_action=all_opportunities_validated next_action=morning_report_delivered
[2026-09-21 22:54:41] [test_sess] [DISCOVERY] [EVENT=CHECKPOINT] [⚠️ [GUARD_INTERCEPT] Intercepted attempt to set phase='complete' while overnight budget remains (0.0m / 240.0m). Mandating Rediscovery Phase instead of early exit.]
[2026-09-21 22:54:41] [session_night_20260921_accel] [GRAPH] [EVENT=EXTERNAL_RESEARCH] opportunity=sample_op decision=hybrid sources=2
[2026-09-21 22:54:41] [test_sess_deadline] [RECON] [EVENT=SESSION_START] session=test_sess_deadline overnight=True budget=180m deadline=2026-09-21T13:54:41.548091+00:00 mainline=none secondary=none
[2026-09-22 10:11:01] [session_night_20260922_runpod_e2e] [REDISCOVERY] [EVENT=SESSION_START] session=session_night_20260922_runpod_e2e overnight=True budget=240m deadline=2026-09-22T02:11:01.091358+00:00 mainline=none secondary=none
[2026-09-22 10:14:27] [session_night_20260922_runpod_e2e] [GRAPH] [EVENT=EXTERNAL_RESEARCH] opportunity=detection_tracking_pareto_evaluator decision=adapted sources=3
[2026-09-22 10:14:33] [session_night_20260922_runpod_e2e] [GRAPH] [EVENT=EXTERNAL_RESEARCH] opportunity=e2e_video_detection_profiler decision=adapted sources=2
[2026-09-22 10:14:39] [session_night_20260922_runpod_e2e] [REDISCOVERY] [EVENT=CHECKPOINT] RunPod 24GB NVIDIA GPU 14-min full video E2E verified. Proved SAMURAI GPU contention (22 FPS vs 183 FPS peak post-SAMURAI), effective batch size 50 frames, 395 player tracklets over 25,316 frames (1 IDSW/2.1s), 70 pass events (5.0/min), and 67s synchronous Supabase PATCH overhead.
[2026-09-22 10:19:42] [session_night_20260922_runpod_e2e] [GRAPH] [EVENT=BRANCH_ACTIVATED] tier=main opportunity=samurai_bounded_memory_pool
[2026-09-22 10:19:42] [session_night_20260922_runpod_e2e] [EXPLORING] [EVENT=STATE_SYNC] opportunity=samurai_bounded_memory_pool mainline=samurai_bounded_memory_pool secondary=none phase=exploring
[2026-09-22 10:19:42] [session_night_20260922_runpod_e2e] [GRAPH] [EVENT=OPPORTUNITY_TRANSITION] opportunity=samurai_bounded_memory_pool from=PARKED to=ACTIVATED
[2026-09-22 10:19:46] [session_night_20260922_runpod_e2e] [GRAPH] [EVENT=BRANCH_ACTIVATED] tier=secondary opportunity=supabase_rpc_merge_session_extra_fix
[2026-09-22 10:19:46] [session_night_20260922_runpod_e2e] [EXPLORING] [EVENT=STATE_SYNC] opportunity=supabase_rpc_merge_session_extra_fix mainline=samurai_bounded_memory_pool secondary=supabase_rpc_merge_session_extra_fix phase=exploring
[2026-09-22 10:19:46] [session_night_20260922_runpod_e2e] [GRAPH] [EVENT=OPPORTUNITY_TRANSITION] opportunity=supabase_rpc_merge_session_extra_fix from=PARKED to=ACTIVATED
[2026-09-22 10:19:50] [session_night_20260922_runpod_e2e] [EXPLORING] [EVENT=STATE_SYNC] opportunity=samurai_bounded_memory_pool mainline=samurai_bounded_memory_pool secondary=supabase_rpc_merge_session_extra_fix phase=exploring
[2026-09-22 10:19:50] [session_night_20260922_runpod_e2e] [GRAPH] [EVENT=OPPORTUNITY_TRANSITION] opportunity=samurai_bounded_memory_pool from=ACTIVATED to=EXPLORING
[2026-09-22 10:19:50] [session_night_20260922_runpod_e2e] [EXPLORING] [EVENT=STATE_SYNC] opportunity=supabase_rpc_merge_session_extra_fix mainline=samurai_bounded_memory_pool secondary=supabase_rpc_merge_session_extra_fix phase=exploring
[2026-09-22 10:19:50] [session_night_20260922_runpod_e2e] [GRAPH] [EVENT=OPPORTUNITY_TRANSITION] opportunity=supabase_rpc_merge_session_extra_fix from=ACTIVATED to=EXPLORING
[2026-09-22 10:20:05] [session_night_20260922_runpod_e2e] [GRAPH] [EVENT=EXTERNAL_RESEARCH] opportunity=samurai_bounded_memory_pool decision=adapted sources=2
[2026-09-22 10:20:09] [session_night_20260922_runpod_e2e] [GRAPH] [EVENT=EXTERNAL_RESEARCH] opportunity=supabase_rpc_merge_session_extra_fix decision=adapted sources=2
[2026-09-22 10:20:27] [test_finalize_sess] [EXPLORING] [EVENT=SESSION_END] Overnight mode ended explicitly.
[2026-09-22 10:20:27] [test_finalize_sess] [COMPLETED] [EVENT=SESSION_FINALIZED] Session finalized via canonical shutdown sequence.
[2026-09-22 10:20:27] [test_sess] [RECON] [EVENT=SESSION_END] Overnight mode ended explicitly.
[2026-09-22 10:20:27] [test_sess] [COMPLETE] [EVENT=CHECKPOINT] current_action=all_opportunities_validated next_action=morning_report_delivered
[2026-09-22 10:20:27] [test_sess] [DISCOVERY] [EVENT=CHECKPOINT] [⚠️ [GUARD_INTERCEPT] Intercepted attempt to set phase='complete' while overnight budget remains (0.0m / 240.0m). Mandating Rediscovery Phase instead of early exit.]
[2026-09-22 10:20:27] [session_night_20260922_runpod_e2e] [GRAPH] [EVENT=EXTERNAL_RESEARCH] opportunity=sample_op decision=hybrid sources=2
[2026-09-22 10:20:27] [test_sess_deadline] [RECON] [EVENT=SESSION_START] session=test_sess_deadline overnight=True budget=180m deadline=2026-09-22T01:20:27.883073+00:00 mainline=samurai_bounded_memory_pool secondary=supabase_rpc_merge_session_extra_fix
[2026-09-22 10:33:38] [session_night_20260922_runpod_e2e] [GRAPH] [EVENT=BRANCH_DEACTIVATED] tier=main opportunity=samurai_bounded_memory_pool status=VALIDATED_SANDBOX
[2026-09-22 10:33:38] [session_night_20260922_runpod_e2e] [EXPLORING] [EVENT=STATE_SYNC] opportunity=samurai_bounded_memory_pool mainline=none secondary=supabase_rpc_merge_session_extra_fix phase=exploring
[2026-09-22 10:33:38] [session_night_20260922_runpod_e2e] [GRAPH] [EVENT=OPPORTUNITY_TRANSITION] opportunity=samurai_bounded_memory_pool from=EXPLORING to=VALIDATED_SANDBOX
[2026-09-22 10:33:38] [session_night_20260922_runpod_e2e] [EXPLORING] [EVENT=STATE_SYNC] opportunity=samurai_bounded_memory_pool mainline=none secondary=supabase_rpc_merge_session_extra_fix phase=exploring
[2026-09-22 10:33:38] [session_night_20260922_runpod_e2e] [GRAPH] [EVENT=OPPORTUNITY_TRANSITION] opportunity=samurai_bounded_memory_pool from=VALIDATED_SANDBOX to=PRODUCTION_CANDIDATE
[2026-09-22 10:33:43] [session_night_20260922_runpod_e2e] [GRAPH] [EVENT=BRANCH_DEACTIVATED] tier=secondary opportunity=supabase_rpc_merge_session_extra_fix status=VALIDATED_SANDBOX
[2026-09-22 10:33:43] [session_night_20260922_runpod_e2e] [REDISCOVERY] [EVENT=STATE_SYNC] opportunity=supabase_rpc_merge_session_extra_fix mainline=none secondary=none phase=rediscovery
[2026-09-22 10:33:43] [session_night_20260922_runpod_e2e] [GRAPH] [EVENT=OPPORTUNITY_TRANSITION] opportunity=supabase_rpc_merge_session_extra_fix from=EXPLORING to=VALIDATED_SANDBOX
[2026-09-22 10:33:43] [session_night_20260922_runpod_e2e] [REDISCOVERY] [EVENT=STATE_SYNC] opportunity=supabase_rpc_merge_session_extra_fix mainline=none secondary=none phase=rediscovery
[2026-09-22 10:33:43] [session_night_20260922_runpod_e2e] [GRAPH] [EVENT=OPPORTUNITY_TRANSITION] opportunity=supabase_rpc_merge_session_extra_fix from=VALIDATED_SANDBOX to=PRODUCTION_CANDIDATE
[2026-09-22 10:33:48] [test_finalize_sess] [EXPLORING] [EVENT=SESSION_END] Overnight mode ended explicitly.
[2026-09-22 10:33:48] [test_finalize_sess] [COMPLETED] [EVENT=SESSION_FINALIZED] Session finalized via canonical shutdown sequence.
[2026-09-22 10:33:48] [test_sess] [RECON] [EVENT=SESSION_END] Overnight mode ended explicitly.
[2026-09-22 10:33:48] [test_sess] [COMPLETE] [EVENT=CHECKPOINT] current_action=all_opportunities_validated next_action=morning_report_delivered
[2026-09-22 10:33:48] [test_sess] [DISCOVERY] [EVENT=CHECKPOINT] [⚠️ [GUARD_INTERCEPT] Intercepted attempt to set phase='complete' while overnight budget remains (0.0m / 240.0m). Mandating Rediscovery Phase instead of early exit.]
[2026-09-22 10:33:48] [session_night_20260922_runpod_e2e] [GRAPH] [EVENT=EXTERNAL_RESEARCH] opportunity=sample_op decision=hybrid sources=2
[2026-09-22 10:33:48] [test_sess_deadline] [RECON] [EVENT=SESSION_START] session=test_sess_deadline overnight=True budget=180m deadline=2026-09-22T01:33:48.684277+00:00 mainline=none secondary=none
[2026-09-22 11:02:34] [session_night_20260922_runpod_e2e] [REDISCOVERY] [EVENT=STATE_SYNC] opportunity=samurai_bounded_memory_pool mainline=none secondary=none phase=rediscovery
[2026-09-22 11:02:34] [session_night_20260922_runpod_e2e] [GRAPH] [EVENT=OPPORTUNITY_TRANSITION] opportunity=samurai_bounded_memory_pool from=PRODUCTION_CANDIDATE to=VALIDATED_SANDBOX
[2026-09-22 11:02:34] [session_night_20260922_runpod_e2e] [REDISCOVERY] [EVENT=STATE_SYNC] opportunity=supabase_rpc_merge_session_extra_fix mainline=none secondary=none phase=rediscovery
[2026-09-22 11:02:34] [session_night_20260922_runpod_e2e] [GRAPH] [EVENT=OPPORTUNITY_TRANSITION] opportunity=supabase_rpc_merge_session_extra_fix from=PRODUCTION_CANDIDATE to=VALIDATED_SANDBOX
[2026-09-22 17:57:42] [session_night_20260922_runpod_e2e] [REDISCOVERY] [EVENT=STATE_SYNC] opportunity=supabase_rpc_merge_session_extra_fix mainline=none secondary=none phase=rediscovery
[2026-09-22 17:57:42] [session_night_20260922_runpod_e2e] [GRAPH] [EVENT=OPPORTUNITY_TRANSITION] opportunity=supabase_rpc_merge_session_extra_fix from=VALIDATED_SANDBOX to=PRODUCTION_CANDIDATE
[2026-09-22 17:57:49] [test_finalize_sess] [EXPLORING] [EVENT=SESSION_END] Overnight mode ended explicitly.
[2026-09-22 17:57:49] [test_finalize_sess] [COMPLETED] [EVENT=SESSION_FINALIZED] Session finalized via canonical shutdown sequence.
[2026-09-22 17:57:49] [test_sess] [RECON] [EVENT=SESSION_END] Overnight mode ended explicitly.
[2026-09-22 17:57:49] [test_sess] [COMPLETE] [EVENT=CHECKPOINT] current_action=all_opportunities_validated next_action=morning_report_delivered
[2026-09-22 17:57:49] [test_sess] [DISCOVERY] [EVENT=CHECKPOINT] [⚠️ [GUARD_INTERCEPT] Intercepted attempt to set phase='complete' while overnight budget remains (0.0m / 240.0m). Mandating Rediscovery Phase instead of early exit.]
[2026-09-22 17:57:49] [session_night_20260922_runpod_e2e] [GRAPH] [EVENT=EXTERNAL_RESEARCH] opportunity=sample_op decision=hybrid sources=2
[2026-09-22 17:57:49] [test_sess_deadline] [RECON] [EVENT=SESSION_START] session=test_sess_deadline overnight=True budget=180m deadline=2026-09-22T08:57:49.396303+00:00 mainline=none secondary=none
[2026-09-22 17:58:27] [session_night_20260922_runpod_e2e] [REDISCOVERY] [EVENT=CHECKPOINT] Supabase 4-way empirical benchmark verified on live DB
[2026-09-22 18:19:35] [session_night_20260922_runpod_e2e] [REDISCOVERY] [EVENT=STATE_SYNC] opportunity=supabase_rpc_merge_session_extra_fix mainline=none secondary=none phase=rediscovery
[2026-09-22 18:19:35] [session_night_20260922_runpod_e2e] [GRAPH] [EVENT=OPPORTUNITY_TRANSITION] opportunity=supabase_rpc_merge_session_extra_fix from=PRODUCTION_CANDIDATE to=VALIDATED_SANDBOX
[2026-09-22 18:21:17] [test_finalize_sess] [EXPLORING] [EVENT=SESSION_END] Overnight mode ended explicitly.
[2026-09-22 18:21:17] [test_finalize_sess] [COMPLETED] [EVENT=SESSION_FINALIZED] Session finalized via canonical shutdown sequence.
[2026-09-22 18:21:17] [test_sess] [RECON] [EVENT=SESSION_END] Overnight mode ended explicitly.
[2026-09-22 18:21:17] [test_sess] [COMPLETE] [EVENT=CHECKPOINT] current_action=all_opportunities_validated next_action=morning_report_delivered
[2026-09-22 18:21:17] [test_sess] [DISCOVERY] [EVENT=CHECKPOINT] [⚠️ [GUARD_INTERCEPT] Intercepted attempt to set phase='complete' while overnight budget remains (0.0m / 240.0m). Mandating Rediscovery Phase instead of early exit.]
[2026-09-22 18:21:17] [session_night_20260922_runpod_e2e] [GRAPH] [EVENT=EXTERNAL_RESEARCH] opportunity=sample_op decision=hybrid sources=2
[2026-09-22 18:21:17] [test_sess_deadline] [RECON] [EVENT=SESSION_START] session=test_sess_deadline overnight=True budget=180m deadline=2026-09-22T09:21:17.140999+00:00 mainline=none secondary=none
[2026-09-22 18:32:32] [test_finalize_sess] [EXPLORING] [EVENT=SESSION_END] Overnight mode ended explicitly.
[2026-09-22 18:32:32] [test_finalize_sess] [COMPLETED] [EVENT=SESSION_FINALIZED] Session finalized via canonical shutdown sequence.
[2026-09-22 18:32:32] [test_sess] [RECON] [EVENT=SESSION_END] Overnight mode ended explicitly.
[2026-09-22 18:32:32] [test_sess] [COMPLETE] [EVENT=CHECKPOINT] current_action=all_opportunities_validated next_action=morning_report_delivered
[2026-09-22 18:32:32] [test_sess] [DISCOVERY] [EVENT=CHECKPOINT] [⚠️ [GUARD_INTERCEPT] Intercepted attempt to set phase='complete' while overnight budget remains (0.0m / 240.0m). Mandating Rediscovery Phase instead of early exit.]
[2026-09-22 18:32:32] [session_night_20260922_runpod_e2e] [GRAPH] [EVENT=EXTERNAL_RESEARCH] opportunity=sample_op decision=hybrid sources=2
[2026-09-22 18:32:32] [test_sess_deadline] [RECON] [EVENT=SESSION_START] session=test_sess_deadline overnight=True budget=180m deadline=2026-09-22T09:32:32.327474+00:00 mainline=none secondary=none
[2026-09-22 09:31:06] [session_night_20260922_runpod_e2e] [REDISCOVERY] [EVENT=CHECKPOINT] Completed STEP 0 36-min gap audit: verified worker idle in pool, no AI summary on GPU worker.
[2026-09-22 09:31:06] [session_night_20260922_runpod_e2e] [GRAPH] [EVENT=OPPORTUNITY_CREATED] opportunity=samurai_concurrency_matrix_benchmark status=VALIDATED_SANDBOX
[2026-09-22 09:31:06] [session_night_20260922_runpod_e2e] [REDISCOVERY] [EVENT=CHECKPOINT] Author and verify bench_samurai_concurrency_matrix.py (22 tests passing). Awaiting RunPod execution.
[2026-09-22 22:45:36] [session_night_20260922_runpod_e2e] [GRAPH] [EVENT=EXTERNAL_RESEARCH] opportunity=sample_op decision=hybrid sources=2
[2026-09-22 22:45:36] [test_sess] [RECON] [EVENT=SESSION_END] Overnight mode ended explicitly.
[2026-09-22 22:45:36] [test_sess] [COMPLETE] [EVENT=CHECKPOINT] current_action=all_opportunities_validated next_action=morning_report_delivered
[2026-09-22 22:45:36] [test_sess] [DISCOVERY] [EVENT=CHECKPOINT] [⚠️ [GUARD_INTERCEPT] Intercepted attempt to set phase='complete' while overnight budget remains (0.0m / 240.0m). Mandating Rediscovery Phase instead of early exit.]
[2026-09-22 22:45:36] [test_sess_deadline] [RECON] [EVENT=SESSION_START] session=test_sess_deadline overnight=True budget=180m deadline=2026-09-22T13:45:36.134163+00:00 mainline=none secondary=none
[2026-09-22 22:45:36] [test_finalize_sess] [EXPLORING] [EVENT=SESSION_END] Overnight mode ended explicitly.
[2026-09-22 22:45:36] [test_finalize_sess] [COMPLETED] [EVENT=SESSION_FINALIZED] Session finalized via canonical shutdown sequence.
[2026-09-22 22:46:16] [session_night_20260922_runpod_e2e] [GRAPH] [EVENT=EXTERNAL_RESEARCH] opportunity=sample_op decision=hybrid sources=2
[2026-09-22 22:46:16] [test_sess] [RECON] [EVENT=SESSION_END] Overnight mode ended explicitly.
[2026-09-22 22:46:16] [test_sess] [COMPLETE] [EVENT=CHECKPOINT] current_action=all_opportunities_validated next_action=morning_report_delivered
[2026-09-22 22:46:16] [test_sess] [DISCOVERY] [EVENT=CHECKPOINT] [⚠️ [GUARD_INTERCEPT] Intercepted attempt to set phase='complete' while overnight budget remains (0.0m / 240.0m). Mandating Rediscovery Phase instead of early exit.]
[2026-09-22 22:46:16] [test_sess_deadline] [RECON] [EVENT=SESSION_START] session=test_sess_deadline overnight=True budget=180m deadline=2026-09-22T13:46:16.538558+00:00 mainline=none secondary=none
[2026-09-22 22:46:16] [test_finalize_sess] [EXPLORING] [EVENT=SESSION_END] Overnight mode ended explicitly.
[2026-09-22 22:46:16] [test_finalize_sess] [COMPLETED] [EVENT=SESSION_FINALIZED] Session finalized via canonical shutdown sequence.
[2026-09-22 22:46:25] [session_night_20260922_runpod_e2e] [GRAPH] [EVENT=EXTERNAL_RESEARCH] opportunity=sample_op decision=hybrid sources=2
[2026-09-22 22:46:25] [test_sess] [RECON] [EVENT=SESSION_END] Overnight mode ended explicitly.
[2026-09-22 22:46:25] [test_sess] [COMPLETE] [EVENT=CHECKPOINT] current_action=all_opportunities_validated next_action=morning_report_delivered
[2026-09-22 22:46:25] [test_sess] [DISCOVERY] [EVENT=CHECKPOINT] [⚠️ [GUARD_INTERCEPT] Intercepted attempt to set phase='complete' while overnight budget remains (0.0m / 240.0m). Mandating Rediscovery Phase instead of early exit.]
[2026-09-22 22:46:25] [test_sess_deadline] [RECON] [EVENT=SESSION_START] session=test_sess_deadline overnight=True budget=180m deadline=2026-09-22T13:46:25.562646+00:00 mainline=none secondary=none
[2026-09-22 22:46:25] [test_finalize_sess] [EXPLORING] [EVENT=SESSION_END] Overnight mode ended explicitly.
[2026-09-22 22:46:25] [test_finalize_sess] [COMPLETED] [EVENT=SESSION_FINALIZED] Session finalized via canonical shutdown sequence.
[2026-09-22 22:48:21] [session_night_20260923] [REDISCOVERY] [EVENT=SESSION_START] session=session_night_20260923 overnight=True budget=240m deadline=2026-09-22T14:48:21.803364+00:00 mainline=none secondary=none
[2026-09-22 22:48:37] [session_night_20260923] [REDISCOVERY] [EVENT=STATE_RECONCILE] Reconciled active state with opportunity graph.
[2026-09-22 22:50:10] [session_night_20260923] [GRAPH] [EVENT=OPPORTUNITY_CREATED] opportunity=expected_threat_xt_engine status=PARKED
[2026-09-22 22:50:15] [session_night_20260923] [GRAPH] [EVENT=EXTERNAL_RESEARCH] opportunity=expected_threat_xt_engine decision=adopted sources=2
[2026-09-22 22:50:19] [session_night_20260923] [GRAPH] [EVENT=OPPORTUNITY_CREATED] opportunity=metabolic_power_fatigue_engine status=PARKED
[2026-09-22 22:50:25] [session_night_20260923] [GRAPH] [EVENT=EXTERNAL_RESEARCH] opportunity=metabolic_power_fatigue_engine decision=adopted sources=2
[2026-09-22 22:50:30] [session_night_20260923] [GRAPH] [EVENT=BRANCH_ACTIVATED] tier=main opportunity=expected_threat_xt_engine
[2026-09-22 22:50:30] [session_night_20260923] [EXPLORING] [EVENT=STATE_SYNC] opportunity=expected_threat_xt_engine mainline=expected_threat_xt_engine secondary=none phase=exploring
[2026-09-22 22:50:30] [session_night_20260923] [GRAPH] [EVENT=OPPORTUNITY_TRANSITION] opportunity=expected_threat_xt_engine from=PARKED to=ACTIVATED
[2026-09-22 22:50:34] [session_night_20260923] [GRAPH] [EVENT=BRANCH_ACTIVATED] tier=secondary opportunity=metabolic_power_fatigue_engine
[2026-09-22 22:50:34] [session_night_20260923] [EXPLORING] [EVENT=STATE_SYNC] opportunity=metabolic_power_fatigue_engine mainline=expected_threat_xt_engine secondary=metabolic_power_fatigue_engine phase=exploring
[2026-09-22 22:50:34] [session_night_20260923] [GRAPH] [EVENT=OPPORTUNITY_TRANSITION] opportunity=metabolic_power_fatigue_engine from=PARKED to=ACTIVATED
[2026-09-22 22:50:38] [session_night_20260923] [EXPLORING] [EVENT=STATE_SYNC] opportunity=expected_threat_xt_engine mainline=expected_threat_xt_engine secondary=metabolic_power_fatigue_engine phase=exploring
[2026-09-22 22:50:38] [session_night_20260923] [GRAPH] [EVENT=OPPORTUNITY_TRANSITION] opportunity=expected_threat_xt_engine from=ACTIVATED to=EXPLORING
[2026-09-22 22:50:43] [session_night_20260923] [EXPLORING] [EVENT=CHECKPOINT] Implementing standalone Expected Threat (xT) engine with 16x12 Markov value iteration
[2026-09-22 22:54:13] [session_night_20260923] [GRAPH] [EVENT=BRANCH_DEACTIVATED] tier=main opportunity=expected_threat_xt_engine status=VALIDATED_SANDBOX
[2026-09-22 22:54:13] [session_night_20260923] [EXPLORING] [EVENT=STATE_SYNC] opportunity=expected_threat_xt_engine mainline=none secondary=metabolic_power_fatigue_engine phase=exploring
[2026-09-22 22:54:13] [session_night_20260923] [GRAPH] [EVENT=OPPORTUNITY_TRANSITION] opportunity=expected_threat_xt_engine from=EXPLORING to=VALIDATED_SANDBOX
[2026-09-22 22:54:44] [session_night_20260923] [EXPLORING] [EVENT=STATE_SYNC] opportunity=metabolic_power_fatigue_engine mainline=none secondary=metabolic_power_fatigue_engine phase=exploring
[2026-09-22 22:54:44] [session_night_20260923] [GRAPH] [EVENT=OPPORTUNITY_TRANSITION] opportunity=metabolic_power_fatigue_engine from=ACTIVATED to=EXPLORING
[2026-09-22 22:54:50] [session_night_20260923] [EXPLORING] [EVENT=CHECKPOINT] Exploring Osgnach 2010 Metabolic Power and High Metabolic Load Distance engine
[2026-09-22 22:56:27] [session_night_20260923] [GRAPH] [EVENT=BRANCH_DEACTIVATED] tier=secondary opportunity=metabolic_power_fatigue_engine status=VALIDATED_SANDBOX
[2026-09-22 22:56:27] [session_night_20260923] [REDISCOVERY] [EVENT=STATE_SYNC] opportunity=metabolic_power_fatigue_engine mainline=none secondary=none phase=rediscovery
[2026-09-22 22:56:27] [session_night_20260923] [GRAPH] [EVENT=OPPORTUNITY_TRANSITION] opportunity=metabolic_power_fatigue_engine from=EXPLORING to=VALIDATED_SANDBOX
[2026-09-22 22:56:44] [session_night_20260923] [REDISCOVERY] [EVENT=STATE_RECONCILE] Reconciled active state with opportunity graph.
[2026-09-22 22:56:57] [session_night_20260923] [GRAPH] [EVENT=EXTERNAL_RESEARCH] opportunity=sample_op decision=hybrid sources=2
[2026-09-22 22:56:57] [test_sess] [RECON] [EVENT=SESSION_END] Overnight mode ended explicitly.
[2026-09-22 22:56:57] [test_sess] [COMPLETE] [EVENT=CHECKPOINT] current_action=all_opportunities_validated next_action=morning_report_delivered
[2026-09-22 22:56:57] [test_sess] [DISCOVERY] [EVENT=CHECKPOINT] [⚠️ [GUARD_INTERCEPT] Intercepted attempt to set phase='complete' while overnight budget remains (0.0m / 240.0m). Mandating Rediscovery Phase instead of early exit.]
[2026-09-22 22:56:57] [test_sess_deadline] [RECON] [EVENT=SESSION_START] session=test_sess_deadline overnight=True budget=180m deadline=2026-09-22T13:56:57.382367+00:00 mainline=none secondary=none
[2026-09-22 22:56:57] [test_finalize_sess] [EXPLORING] [EVENT=SESSION_END] Overnight mode ended explicitly.
[2026-09-22 22:56:57] [test_finalize_sess] [COMPLETED] [EVENT=SESSION_FINALIZED] Session finalized via canonical shutdown sequence.
[2026-09-22 22:58:57] [session_night_20260923] [GRAPH] [EVENT=OPPORTUNITY_CREATED] opportunity=dynamic_formation_tactical_line_analyzer status=PARKED
[2026-09-22 22:59:05] [session_night_20260923] [GRAPH] [EVENT=EXTERNAL_RESEARCH] opportunity=dynamic_formation_tactical_line_analyzer decision=adopted sources=2
[2026-09-22 22:59:12] [session_night_20260923] [GRAPH] [EVENT=BRANCH_SWITCHED] tier=main from=expected_threat_xt_engine to=dynamic_formation_tactical_line_analyzer
[2026-09-22 22:59:12] [session_night_20260923] [EXPLORING] [EVENT=STATE_SYNC] opportunity=dynamic_formation_tactical_line_analyzer mainline=dynamic_formation_tactical_line_analyzer secondary=none phase=exploring
[2026-09-22 22:59:12] [session_night_20260923] [GRAPH] [EVENT=OPPORTUNITY_TRANSITION] opportunity=dynamic_formation_tactical_line_analyzer from=PARKED to=ACTIVATED
[2026-09-22 22:59:19] [session_night_20260923] [EXPLORING] [EVENT=STATE_SYNC] opportunity=dynamic_formation_tactical_line_analyzer mainline=dynamic_formation_tactical_line_analyzer secondary=none phase=exploring
[2026-09-22 22:59:19] [session_night_20260923] [GRAPH] [EVENT=OPPORTUNITY_TRANSITION] opportunity=dynamic_formation_tactical_line_analyzer from=ACTIVATED to=EXPLORING
[2026-09-22 22:59:24] [session_night_20260923] [EXPLORING] [EVENT=CHECKPOINT] Implementing dynamic tactical formation clustering and inter-line spacing analyzer
[2026-09-22 23:00:10] [session_night_20260923] [GRAPH] [EVENT=BRANCH_DEACTIVATED] tier=main opportunity=dynamic_formation_tactical_line_analyzer status=VALIDATED_SANDBOX
[2026-09-22 23:00:10] [session_night_20260923] [REDISCOVERY] [EVENT=STATE_SYNC] opportunity=dynamic_formation_tactical_line_analyzer mainline=none secondary=none phase=rediscovery
[2026-09-22 23:00:10] [session_night_20260923] [GRAPH] [EVENT=OPPORTUNITY_TRANSITION] opportunity=dynamic_formation_tactical_line_analyzer from=EXPLORING to=VALIDATED_SANDBOX
[2026-09-22 23:00:31] [session_night_20260923] [REDISCOVERY] [EVENT=STATE_RECONCILE] Reconciled active state with opportunity graph.
[2026-09-23 13:44:14] [test_finalize_sess] [EXPLORING] [EVENT=SESSION_END] Overnight mode ended explicitly.
[2026-09-23 13:44:14] [test_finalize_sess] [COMPLETED] [EVENT=SESSION_FINALIZED] Session finalized via canonical shutdown sequence.
[2026-09-23 13:44:14] [test_sess] [RECON] [EVENT=SESSION_END] Overnight mode ended explicitly.
[2026-09-23 13:44:14] [test_sess] [COMPLETE] [EVENT=CHECKPOINT] current_action=all_opportunities_validated next_action=morning_report_delivered
[2026-09-23 13:44:14] [test_sess] [DISCOVERY] [EVENT=CHECKPOINT] [⚠️ [GUARD_INTERCEPT] Intercepted attempt to set phase='complete' while overnight budget remains (0.0m / 240.0m). Mandating Rediscovery Phase instead of early exit.]
[2026-09-23 13:44:14] [session_night_20260923] [GRAPH] [EVENT=EXTERNAL_RESEARCH] opportunity=sample_op decision=hybrid sources=2
[2026-09-23 13:44:14] [test_sess_deadline] [RECON] [EVENT=SESSION_START] session=test_sess_deadline overnight=True budget=180m deadline=2026-09-23T04:44:14.887354+00:00 mainline=none secondary=none
[2026-09-23 22:50:16] [session_night_20260923] [GRAPH] [EVENT=OPPORTUNITY_CREATED] opportunity=worker_idempotency_lifecycle_guard status=PARKED
[2026-09-23 22:50:23] [session_night_20260923] [GRAPH] [EVENT=OPPORTUNITY_CREATED] opportunity=pipeline_speed_accuracy_evaluator status=PARKED
[2026-09-23 22:50:32] [session_night_20260923] [GRAPH] [EVENT=BRANCH_SWITCHED] tier=main from=dynamic_formation_tactical_line_analyzer to=worker_idempotency_lifecycle_guard
[2026-09-23 22:50:32] [session_night_20260923] [EXPLORING] [EVENT=STATE_SYNC] opportunity=worker_idempotency_lifecycle_guard mainline=worker_idempotency_lifecycle_guard secondary=none phase=exploring
[2026-09-23 22:50:32] [session_night_20260923] [GRAPH] [EVENT=OPPORTUNITY_TRANSITION] opportunity=worker_idempotency_lifecycle_guard from=PARKED to=ACTIVATED
[2026-09-23 22:50:38] [session_night_20260923] [GRAPH] [EVENT=BRANCH_SWITCHED] tier=secondary from=metabolic_power_fatigue_engine to=pipeline_speed_accuracy_evaluator
[2026-09-23 22:50:38] [session_night_20260923] [EXPLORING] [EVENT=STATE_SYNC] opportunity=pipeline_speed_accuracy_evaluator mainline=worker_idempotency_lifecycle_guard secondary=pipeline_speed_accuracy_evaluator phase=exploring
[2026-09-23 22:50:38] [session_night_20260923] [GRAPH] [EVENT=OPPORTUNITY_TRANSITION] opportunity=pipeline_speed_accuracy_evaluator from=PARKED to=ACTIVATED
[2026-09-23 22:50:47] [session_night_20260923] [EXPLORING] [EVENT=STATE_SYNC] opportunity=worker_idempotency_lifecycle_guard mainline=worker_idempotency_lifecycle_guard secondary=pipeline_speed_accuracy_evaluator phase=exploring
[2026-09-23 22:50:47] [session_night_20260923] [GRAPH] [EVENT=OPPORTUNITY_TRANSITION] opportunity=worker_idempotency_lifecycle_guard from=ACTIVATED to=EXPLORING
[2026-09-23 23:05:00] [session_night_20260923] [EXPLORING] [EVENT=CHECKPOINT] Authored test_worker_idempotency_lifecycle.py covering 5-layer reproduction & regression (refresh, back-navigation, gateway idempotency, handler pre-download check, history state sanitization). 5/5 tests passed in 0.001s.
[2026-09-23 23:08:00] [session_night_20260923] [GRAPH] [EVENT=OPPORTUNITY_TRANSITION] opportunity=worker_idempotency_lifecycle_guard from=EXPLORING to=PRODUCTION_CANDIDATE
[2026-09-23 23:08:00] [session_night_20260923] [EXPLORING] [EVENT=STATE_SYNC] opportunity=worker_idempotency_lifecycle_guard mainline=worker_idempotency_lifecycle_guard secondary=pipeline_speed_accuracy_evaluator phase=exploring
[2026-09-23 23:10:00] [session_night_20260923] [GRAPH] [EVENT=OPPORTUNITY_TRANSITION] opportunity=pipeline_speed_accuracy_evaluator from=ACTIVATED to=EXPLORING
[2026-09-23 23:10:00] [session_night_20260923] [CONSOLIDATION] [EVENT=STATE_SYNC] opportunity=pipeline_speed_accuracy_evaluator mainline=worker_idempotency_lifecycle_guard secondary=pipeline_speed_accuracy_evaluator phase=consolidation
[2026-09-23 23:10:00] [session_night_20260923] [CONSOLIDATION] [EVENT=CHECKPOINT] Profiled RunPod 25k frame execution: 95.6 FPS (~4x real-time). Identified 2 critical accuracy bottlenecks: uniform PLAYER_CONF=0.59 dropping balls, and ByteTrack match_thresh=0.80 under Stride 3.

[2026-09-24 19:12:00] [session_night_20260924_evidence_review] [IDLE] [EVENT=SESSION_START] session=session_night_20260924_evidence_review mainline=samurai_concurrency_matrix_benchmark secondary=supabase_rpc_merge_session_extra_fix
[2026-09-24 19:12:05] [session_night_20260924_evidence_review] [RECON] [EVENT=CHECKPOINT] Resuming Autonomous Expansion: reading 2 RunPod production logs (C=11 OOM crash, C=4 success) for session fc54b442-3891-4cd4-8869-f6efef0d6583
[2026-09-24 19:14:00] [session_night_20260924_evidence_review] [RECON] [EVENT=CHECKPOINT] Read opportunity_graph.json (1764 lines, all nodes), session_log.md, failed_paths.json. Confirmed all opportunities at VALIDATED_SANDBOX. 4 opportunities at INTEGRATED (human-authorized).
[2026-09-24 19:16:00] [session_night_20260924_evidence_review] [EXPLORING] [EVENT=CHECKPOINT] Deep analysis of RunPod logs: Run A (cap=16→11 parallel) OOM exit 137 after 84s. Run B (cap=4→4+4+3 waves) success in 439.4s E2E, 240.5s SAMURAI, 102 FPS avg YOLO.
[2026-09-24 19:18:00] [session_night_20260924_evidence_review] [EXPLORING] [EVENT=CHECKPOINT] Created comprehensive evidence review artifact: YOLO contention FPS curve (38-150 during C=4 waves), pipeline bottleneck taxonomy (4 tiers), comparative C=10 vs C=4 vs C=11 analysis.
[2026-09-24 19:20:00] [session_night_20260924_evidence_review] [EXPLORING] [EVENT=CHECKPOINT] Enhanced bench_samurai_concurrency_matrix.py with per-chunk YOLO FPS instrumentation: contention vs unconstrained phase detection via SAMURAI done timestamp. All 13 tests pass (3 benchmark + 10 scheduler).
[2026-09-24 19:38:00] [session_night_20260924_evidence_review] [DISCOVERY] [EVENT=CHECKPOINT] Mandatory Rediscovery Phase initiated per user directive. Investigated common root causes connecting minimap, heatmap, top speed, and running distance.
[2026-09-24 19:40:00] [session_night_20260924_evidence_review] [GRAPH] [EVENT=OPPORTUNITY_CREATED] opportunity=analytics_accuracy_foundation status=ACTIVATED
[2026-09-24 19:40:30] [session_night_20260924_evidence_review] [EXPLORING] [EVENT=CHECKPOINT] Implemented server/pipeline/analytics_accuracy_foundation.py: TeamIdentityMetrics (purity/flips), PitchHomographyMetrics (bounds/teleports), TrajectoryKinematicsMetrics (speed/accel violations), UnifiedAnalyticsAccuracyScorecard (Holistic Accuracy Index).
[2026-09-24 19:41:00] [session_night_20260924_evidence_review] [GRAPH] [EVENT=OPPORTUNITY_TRANSITION] opportunity=analytics_accuracy_foundation from=ACTIVATED to=VALIDATED_SANDBOX
[2026-09-24 19:41:00] [session_night_20260924_evidence_review] [VALIDATING] [EVENT=CHECKPOINT] 6/6 unit tests passed in test_analytics_accuracy_foundation.py. Full 5-suite regression clean (30/30 passed).
[2026-09-24 19:41:30] [session_night_20260924_evidence_review] [REDISCOVERY] [EVENT=CHECKPOINT] Documented Discoveries #10 (RunPod C=11 OOM vs C=4 wave transition contention) and #11 (Unified 5-Stage Analytics Accuracy Foundation) in discoveries.md.

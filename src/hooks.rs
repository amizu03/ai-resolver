use core::f32::consts::PI;

use crate::neural_network::*;
use crate::{dbg, neural_network::norm_yaw, prelude::*, println};
use num_traits::*;

#[derive(Copy, Clone, Debug)]
#[repr(C)]
struct GlobalVarsBase {
    pad: [u8; 4],
    pub framecount: i32,
    pad1: [u8; 8],
    pub curtime: f32,
}

#[derive(Copy, Clone, Debug)]
#[repr(C)]
struct AnimLayer {
    pad: [u8; 8],
    pub sequence: i32,
    pub prev_cycle: f32,
    pub weight: f32,
    pub weight_delta_rate: f32,
    pub playback_rate: f32,
    pub cycle: f32,
    pad1: [u8; 8],
    pub owner: usize,
    pub invalidate_physics_bits: i32,
    pad2: [u8; 44],
}

static_assertions::const_assert_eq!(size_of::<AnimLayer>(), 0x5C);

#[derive(Copy, Clone, Default, Debug)]
struct PositiveTurnData {
    pub last_turn_time: f32,
    pub turn_right_started: bool,
    pub turn_start_lby: f32,
    pub turn_distance: Option<f32>,
    pub turn_rate: Option<f32>,
}

#[derive(Copy, Clone, Default, Debug)]
struct LbyTimer {
    pub lby_realign_timer: f32,
}

#[derive(Copy, Clone, Default, Debug)]
struct YawRotation {
    pub lby_rotate_time: f32,
    pub spin_start_time: Option<f32>,
    pub rotate_rate: Option<f32>,
}

#[derive(Copy, Clone, Default, Debug)]
struct MotionData {
    pub delta_ticks: f32,
    pub rate: f32,
}

#[derive(Copy, Clone, Default, Debug)]
struct ResolverData {
    // resolver data for neural network
    pub change_lby: Option<f32>,
    pub old_lby: Option<f32>,
    pub old_yaw: Option<f32>,
    pub last_moving_lby: Option<f32>,
    pub lby_change_time: f32,
    pub lby_rate: Option<f32>,
    pub eye_yaw_rate: Option<f32>,
    // resolver data
    pub lby_timer: Option<LbyTimer>,
    pub positive_turn: Option<PositiveTurnData>,
    pub last_positive_turn_time: Option<f32>,
    pub yaw_rotation: Option<YawRotation>,
    pub last_ground_move_time: Option<f32>,
    pub old_animlayers: Option<[AnimLayer; 13]>,
    pub yaw_additive: f32,
    pub yaw_additive_time: Option<f32>,
    pub yaw_additive_time1: Option<f32>,
    pub distortion: Option<MotionData>,
    pub distortion_correction: Option<MotionData>,
    pub rotation: Option<MotionData>,
}

const ACT_CSGO_IDLE_ADJUST_STOPPEDMOVING: i32 = 5;
const ACT_CSGO_IDLE_TURN_BALANCEADJUST: i32 = 4;

static mut RESOLVER_DATA: Option<ResolverData> = None;
static mut RESOLVER: Option<BaseResolver> = None;

unsafe extern "fastcall" fn animstate_update(global_vars: &mut GlobalVarsBase, animstate: usize) {
    let player = *((animstate + 0x50) as *mut usize);

    if player == 0 {
        RESOLVER_DATA = None;
        RESOLVER = None;
        println!("null player");
        return;
    }

    let animlayers: Option<&mut [AnimLayer; 13]> = transmute(*((player + 0x4B0) as *const usize));

    if animlayers.is_none() {
        RESOLVER_DATA = None;
        RESOLVER = None;
        println!("null animlayers");
        return;
    }

    let animlayers = animlayers.unwrap();

    if RESOLVER_DATA.is_none() {
        RESOLVER_DATA = Some(Default::default());
    }

    let data = RESOLVER_DATA.as_mut().unwrap();

    // replace with clientsided dormancy check
    // let dormant = false;
    // if dormant {
    //     RESOLVER_DATA = None;
    // }

    let lby = norm_yaw(*((player + 0x258C) as *mut f32));
    let abs_vel = *((player + 0x178) as *mut Vector3);

    let adjust_started = *((animstate + 0x130) as *mut bool);
    let eye_yaw = norm_yaw(*((animstate + 0x68) as *mut f32));
    let eye_pitch = norm_yaw(*((animstate + 0x6C) as *mut f32));
    // last update increment
    let t = *((animstate + 0x64) as *mut f32);
    let ticks = (t * 64.0 + 0.5) as i32 as f32;

    let speed_2d = abs_vel.xy().magnitude();

    let moving = speed_2d > 0.1;
    let moving_on_ground = moving && abs_vel.z == 0.0;
    let in_air = abs_vel.z != 0.0;
    let on_ground = !in_air;

    // LBY updates consistently when moving and on ground
    // if moving_on_ground {
    //     RESOLVER_DATA.last_moving_lby = lby;
    // }

    // if animlayers[3].sequence != ACT_CSGO_IDLE_TURN_BALANCEADJUST
    //     || animlayers[4].playback_rate != 0.0
    //     || moving
    //     || in_air
    // {
    //     data.positive_turn = None;
    // }

    // record lby changes
    if let Some(old_lby) = data.old_lby {
        if old_lby != lby {
            data.change_lby = Some(old_lby);
            data.lby_rate = Some(
                norm_yaw(lby - old_lby) / ((global_vars.curtime - data.lby_change_time) * 64.0),
            );
            data.lby_change_time = global_vars.curtime;
            // println!("Lby changed");
        }
    }

    let lby_changed = matches!(data.old_lby, Some(old_lby) if lby != old_lby);
    let mut lby_accurate = false;
    let mut time_until_lby_accurate = None;

    // let lby_update =
    // if moving && (animlayers[4].playback_rate == 0.0 || on_ground)
    // replicate lby update timer serverside behaviour in csgo animstate to predict lby updates
    // if moving || abs_vel.z.abs() > 100.0 {
    //     RESOLVER_DATA.lby_realign_timer = global_vars.curtime + 0.22;
    // }
    if on_ground {
        if moving {
            data.last_ground_move_time = Some(global_vars.curtime);
            data.lby_timer = Some(LbyTimer {
                lby_realign_timer: global_vars.curtime + 0.22,
            });
            data.last_moving_lby = Some(lby);
        } else if let Some(lby_timer) = &mut data.lby_timer {
            let realign_time_reached = global_vars.curtime > lby_timer.lby_realign_timer;

            time_until_lby_accurate = Some(lby_timer.lby_realign_timer - global_vars.curtime);

            // println!({animlayers[3].cycle:.3}", "{animlayers[3].weight:.3});
            match (realign_time_reached, lby_changed) {
                (true, false) if matches!(data.old_animlayers, Some(old_animlayers) if animlayers[3].cycle >= old_animlayers[3].cycle) =>
                {
                    // println!("a");
                    // dbg!("a", animlayers[3].cycle, animlayers[3].weight);
                    // println!({global_vars.curtime:.3}" ,"{lby_timer.lby_realign_timer:.3});
                    // println!("Static LBY update");
                    lby_timer.lby_realign_timer = global_vars.curtime + 1.1;
                    lby_accurate = true;
                }
                // (true, false) => {
                //     if let Some(old_animlayers) = data.old_animlayers {
                //         // println!("a");
                //         // dbg!("a", animlayers[3].cycle, animlayers[3].weight);
                //         println!("new: "{animlayers[3].weight:.3}" ,"{animlayers[3].cycle:.3});
                //         println!("old: "{old_animlayers[3].weight:.3}" ,"{old_animlayers[3].cycle:.3}"\n");
                //         lby_timer.lby_realign_timer = global_vars.curtime + 1.1;
                //     }
                // }
                // our lby timer drifted from server timer
                // if lby changes but our timer doesnt activate, there was an lby update and our timer was wrong
                // we correct it now
                (false, true) => {
                    // dbg!("b", animlayers[3].cycle, animlayers[3].weight);
                    // println!("Corrected LBY timer drift: "{global_vars.curtime - lby_timer.lby_realign_timer:.3}"s");
                    lby_timer.lby_realign_timer = global_vars.curtime + 1.1;
                    lby_accurate = true;
                }
                // definitely lby updated
                // difference between foot and eye yaw is at least 35.0
                // from game
                // abs( AngleDiff( m_flFootYaw, m_flEyeYaw ) ) > 35.0f
                (true, true) => {
                    // dbg!("c", animlayers[3].cycle, animlayers[3].weight);
                    // println!("LBY change update");
                    lby_timer.lby_realign_timer = global_vars.curtime + 1.1;
                    lby_accurate = true;
                }
                // lby not updated yet
                (false, false) => {}
                // lby timer reached but we have no update, lby breaker is probably disabled or they have no fake/real antiaim
                (true, false) => {}
            }

            // detect sudden right turns
            if animlayers[3].sequence == ACT_CSGO_IDLE_TURN_BALANCEADJUST {
                let last_turn_delta_time = global_vars.curtime
                    - data.last_positive_turn_time.unwrap_or(global_vars.curtime);

                if let Some(old_animlayers) = data.old_animlayers {
                    // lby updated to the right turn right started
                    if old_animlayers[3].cycle == 0.0 && animlayers[3].cycle > 0.0 {
                        data.last_positive_turn_time = Some(global_vars.curtime);
                        data.positive_turn = Some(PositiveTurnData {
                            turn_distance: None,
                            turn_rate: None,
                            last_turn_time: global_vars.curtime,
                            turn_right_started: true,
                            turn_start_lby: lby,
                        });
                    }
                }

                // detect right turn stopping, count how long it took to rotate that distance to find rotate rate
                if let Some(positive_turn) = &mut data.positive_turn
                    && let Some(lby_rate) = data.lby_rate
                    && matches!(data.old_animlayers, Some(old_animlayers) if old_animlayers[3].sequence == ACT_CSGO_IDLE_TURN_BALANCEADJUST)
                    && positive_turn.turn_right_started
                    && animlayers[3].cycle == 0.0
                {
                    // if we dont have a base turning rate, calculate new one
                    // else add to the total turn
                    match &mut positive_turn.turn_distance {
                        Some(turn_dist) => *turn_dist += lby_rate * ticks,
                        None => {
                            positive_turn.turn_distance = Some(lby_rate * ticks);
                        }
                    }

                    // calculate turning rate
                    let time_until_lby_update = lby_timer.lby_realign_timer - global_vars.curtime;
                    let turn_duration = time_until_lby_update
                        .min(global_vars.curtime - positive_turn.last_turn_time);
                    let turn_duration_ticks = turn_duration * 64.0;
                    positive_turn.turn_rate =
                        Some(positive_turn.turn_distance.unwrap() / turn_duration_ticks);
                    println!("tdist: "{positive_turn.turn_distance.unwrap():.3});
                    println!({positive_turn.turn_rate.unwrap():.3}", "{time_until_lby_update:.3}", "{last_turn_delta_time:.3});

                    let mut distort_time = time_until_lby_update - last_turn_delta_time;
                    if distort_time < 0.0 {
                        distort_time = last_turn_delta_time;
                    }
                    let distort_ticks = distort_time * 64.0;
                    println!("predicted distort: "{positive_turn.turn_rate.unwrap() * dbg!(distort_ticks):.3});
                    println!("base lby real offset: "{positive_turn.turn_rate.unwrap() * (time_until_lby_update * 64.0):.3});
                    // println!("base lby real offset: "{positive_turn.turn_rate.unwrap() * (time_until_lby_update * 64.0):.3});

                    // stop calculating turn because it just ended
                    positive_turn.turn_right_started = false;

                    data.distortion = Some(MotionData {
                        delta_ticks: distort_ticks,
                        rate: positive_turn.turn_rate.unwrap(),
                    });

                    // if < 1 {
                    //     *= 10_000.0
                    // }

                    // data.yaw_additive = 0.0;
                    data.yaw_additive_time = Some(global_vars.curtime);

                    // let distortion_turn_rate = matches!(positive_turn.turn_rate, Some(turn_rate) if turn_rate.abs() > 4.8 && turn_rate.abs() < 8.5);

                    // match positive_turn.turn_rate {
                    //     Some(d) if d.abs() < 16.0 => {
                    //         // println!("Distortion "{positive_turn.turn_rate:.3?});
                    //         let turn_amount =
                    //             positive_turn.turn_rate.unwrap() * turn_duration_ticks;
                    //         let turn_amount_since_lby_update =
                    //             positive_turn.turn_rate.unwrap() * (time_until_lby_update * 64.0);

                    //         if let Some(old_eye_yaw) = data.old_yaw {
                    //             DISTORT_ANGLE = norm_yaw(lby - turn_amount);

                    //             println!("D: "{time_until_lby_update:.3}", "{turn_amount:.3}", "{turn_amount_since_lby_update:.3});
                    //             // println!("D: "{turn_duration:.3}", "{time_until_lby_update:.3}", "{positive_turn.turn_rate.unwrap():0.3}", "{:0.3}", "{norm_yaw(eye_yaw - old_eye_yaw).abs():.3});
                    //         }
                    //     }
                    //     None if turn_duration > t * 5.0
                    //         && turn_duration < time_until_lby_update
                    //         && time_until_lby_update / turn_duration > 1.5 =>
                    //     {
                    //         println!("DETECTED DISTORTION!");
                    //     }
                    //     _ => {}
                    // }
                }

                // println!("DF: "{norm_yaw(eye_yaw - lby):.3});

                // detect rotation antiaims/spin antiaim
                if let Some(change_lby) = data.change_lby
                    && let Some(lby_rate) = data.lby_rate
                    && lby_changed
                {
                    if data.yaw_rotation.is_none() {
                        data.yaw_rotation = Some(YawRotation {
                            lby_rotate_time: global_vars.curtime,
                            spin_start_time: None,
                            rotate_rate: None,
                        });
                    }

                    if let Some(yaw_rotation) = &mut data.yaw_rotation {
                        let old_rotate_rate = yaw_rotation.rotate_rate;
                        yaw_rotation.rotate_rate = Some(lby_rate);

                        let rotated_amount = lby_rate * ticks;
                        let abs_rotate_rate = lby_rate.abs();

                        if let Some(old_rate) = old_rotate_rate {
                            println!("rot: "{abs_rotate_rate:.2}", "{(lby_rate - old_rate).abs()});
                            if abs_rotate_rate > 14.0
                                || ((lby_rate - old_rate).abs() < 0.1 && abs_rotate_rate < 10.0)
                            {
                                let rotate_time = yaw_rotation
                                    .spin_start_time
                                    .map_or(ticks, |t| global_vars.curtime - t);
                                let rotate_ticks = rotate_time * 64.0;
                                let rotate_amount = lby_rate * rotate_ticks;

                                yaw_rotation.spin_start_time = Some(global_vars.curtime);

                                // distortion will have additional rotation always opposing direction
                                if data.distortion.is_some() {
                                    println!("Rotation: "{-lby_rate * ticks:.2});
                                    //     ||
                                    // {
                                    // distortion automatic rotation will take over after we figure out the distort speed and exhaust it
                                    data.distortion_correction = Some(MotionData {
                                        delta_ticks: rotate_ticks,
                                        rate: -lby_rate,
                                    });
                                    // data.yaw_additive = 0.0;
                                    data.yaw_additive_time1 = Some(global_vars.curtime);
                                    // }

                                    // if matches!(data.distortion, Some(distort) if distort.delta_ticks > 0.0)
                                    // {
                                    //     data.distortion_correction.unwrap().delta_ticks
                                    // }
                                }
                            } else {
                                yaw_rotation.spin_start_time = None;
                            }
                        }

                        // if let Some(old_rate) = old_rotate_rate {
                        //     let is_slow_rotation = dbg!(abs_rotate_rate) > 1.0
                        //         && old_rate.abs() > 1.0
                        //         && abs_rotate_rate < 5.0;
                        //     let is_fast_rotation = (dbg!(lby_rate) - old_rate).abs() < 0.01;

                        //     if (is_slow_rotation || is_fast_rotation)
                        //         && lby_rate.signum() == old_rate.signum()
                        //     {
                        //         if is_slow_rotation {
                        //             println!("Slow rotate");
                        //         } else {
                        //             println!("Fast rotate");
                        //         }

                        //         yaw_rotation.spin_start_time = Some(global_vars.curtime);
                        //     }
                        // }
                    }
                }
            }
        }
    } else {
        data.yaw_rotation = None;
        data.change_lby = None;
        data.lby_rate = None;
        data.last_ground_move_time = None;
        data.lby_timer = None;
    }

    // store data

    // train resolver model

    if let Some(yaw_additive_time) = data.yaw_additive_time {
        let ticks_since_distort_motion =
            ((global_vars.curtime - yaw_additive_time) * 64.0) as usize;

        let ticks_since_distort_base_motion =
            ((global_vars.curtime - data.yaw_additive_time.unwrap_or(0.0)) * 64.0) as usize;

        let target_delta = norm_yaw(lby - eye_yaw);
        let target = [
            target_delta.to_radians().sin(),
            target_delta.to_radians().cos(),
        ];

        match (
            &mut data.distortion,
            &mut data.distortion_correction,
            time_until_lby_accurate,
            moving_on_ground,
        ) {
            (Some(distort), Some(rotation), Some(rotate_time), false) => {
                let time_until_next_rotate_cycles =
                    (1.1 - rotate_time.clamp(0.0, 1.1 + (ticks as f32 / 64.0))) / 1.1;
                let distort_fraction = ticks_since_distort_motion as f32 / distort.delta_ticks;
                let rotate_fraction = ticks_since_distort_base_motion as f32 / rotation.delta_ticks;
                let last_distorted_amount = distort.rate * distort.delta_ticks;
                let last_rotate_amount = rotation.rate * distort.delta_ticks;
                let lby_distort = data.lby_rate.unwrap_or(0.0) * distort.delta_ticks;
                let lby_rotate = data.lby_rate.unwrap_or(0.0) * rotation.delta_ticks;

                let mut processed_distort_update = false;
                let inputs = [
                    last_distorted_amount.to_radians().sin(),
                    last_distorted_amount.to_radians().cos(),
                    last_rotate_amount.to_radians().sin(),
                    last_rotate_amount.to_radians().cos(),
                    lby_distort.to_radians().sin(),
                    lby_rotate.to_radians().cos(),
                    time_until_next_rotate_cycles,
                    (time_until_next_rotate_cycles * 2.0 * PI).sin(),
                    distort_fraction,
                    (distort_fraction * (PI / 2.0)).sin(),
                    rotate_fraction,
                    (rotate_fraction * (PI / 2.0)).sin(),
                ];

                let predicted = if RESOLVER.is_none() {
                    RESOLVER = Some(BaseResolver::new());
                    [0.0; 2]
                } else {
                    RESOLVER.as_ref().unwrap().forward(inputs)
                };

                let predicted_offset = predicted[0].atan2(predicted[1]).to_degrees();
                let predicted_yaw = norm_yaw(lby + predicted_offset);
                let delta_target_yaw = norm_yaw(predicted_yaw - eye_yaw);
                if delta_target_yaw.abs() > 35.0 / 4.0 {
                    static mut SKIP: usize = 0;
                    if SKIP >= 3 {
                        SKIP = 0;
                    } else {
                        SKIP += 1;
                    }
                    if (rotate_time > t * 2.0 && SKIP == 3) || lby_changed {
                        RESOLVER.as_mut().unwrap().train_step(inputs, target, 0.05);
                        println!("diff: "{delta_target_yaw:.6});
                    }
                } else {
                    println!("!");
                }

                // apply initial lby-real offset
                data.yaw_additive = if rotation.rate > 0.0 { 120.0 } else { -120.0 };
                let rotate_time = time_until_lby_accurate.unwrap_or(0.0);

                // apply distortion rotation since lby update
                data.yaw_additive += distort.rate * distort.delta_ticks;

                let last_update_time =
                    data.lby_timer.map_or(0.0, |timer| timer.lby_realign_timer) - 1.1;
                if last_update_time > yaw_additive_time {
                    let distort_time_after_lby_reset = global_vars.curtime - last_update_time;
                    let distort_ticks_after_lby_reset = distort_time_after_lby_reset * 64.0;

                    // println!("lby_rate: "{data.lby_rate.unwrap_or(0.0):.3});
                    // data.yaw_additive = if data.lby_rate.unwrap_or(0.0) > 0.0 {
                    //     120.0
                    // } else {
                    //     -120.0
                    // };

                    data.yaw_additive = if data.lby_rate.unwrap_or(0.0) > 0.0 {
                        -120.0
                    } else {
                        120.0
                    };
                    data.yaw_additive -= distort.rate * distort.delta_ticks as f32;
                    // data.yaw_additive = distort.rate * distort.delta_ticks;
                    let mut cycles = 0;
                    let mut last_rotate_time = last_update_time;
                    let mut rotate_time = last_update_time;

                    while rotate_time > yaw_additive_time {
                        let sign_flip = if cycles == 0 || cycles % 2 == 1 {
                            data.lby_rate.unwrap_or(0.0)
                        } else {
                            -data.lby_rate.unwrap_or(0.0)
                        };
                        let dt = rotate_time - last_rotate_time;
                        let dt_ticks = dt * 64.0;
                        data.yaw_additive += rotation.rate * sign_flip * dt_ticks;
                        cycles += 1;

                        if cycles == 0 && global_vars.curtime - yaw_additive_time <= 0.22 {
                            rotate_time -= 0.22;
                        } else {
                            rotate_time -= 1.1;
                        }
                        last_rotate_time = rotate_time;
                    }

                    // println!("diff: "{norm_yaw(norm_yaw(lby + data.yaw_additive) - eye_yaw):.2});
                    // println!("correct: "{norm_yaw(lby - eye_yaw):.2});
                    // } else {
                    // data.yaw_additive += distort.rate * distort_ticks_after_lby_reset;
                    // }
                } else {
                    // let positive_lby = norm_yaw(lby + 120.0);
                    // let negative_lby = norm_yaw(lby - 120.0);
                    // let is_positive_lby = norm_yaw(eye_yaw - positive_lby).abs()
                    //     < norm_yaw(eye_yaw - negative_lby).abs();
                    // let lby_base_offset = if is_positive_lby { 120.0 } else { -120.0 };

                    let dt = yaw_additive_time - last_update_time;
                    if dt > 0.0 && dt <= 0.22 {
                        data.yaw_additive = (2.6 * distort.delta_ticks).copysign(distort.rate);

                        // do distortion rotation until lby updates
                        if ticks_since_distort_motion as f32 / 64.0 < rotate_time {
                            // data.yaw_additive -= distort.rate * distort.delta_ticks;
                            data.yaw_additive += distort.rate * (rotate_time * 64.0);
                        } else {
                            data.yaw_additive += distort.rate * (1.1 * 64.0);
                            data.yaw_additive -= distort.rate
                                * (ticks_since_distort_motion as f32 - distort.delta_ticks);
                        }

                        // println!("diff: "{norm_yaw(norm_yaw(lby + data.yaw_additive) - eye_yaw):.2});
                        // println!("correct: "{norm_yaw(lby - eye_yaw):.2});
                    } else {
                        data.yaw_additive = (2.6 * distort.delta_ticks).copysign(distort.rate);
                        // data.yaw_additive = if distort.rate > 0.0 { 120.0 } else { -120.0 };

                        // do distortion rotation until lby updates
                        if ticks_since_distort_motion as f32 / 64.0 < rotate_time {
                            // data.yaw_additive -= distort.rate * distort.delta_ticks;
                            data.yaw_additive -= distort.rate * ticks_since_distort_motion as f32;

                            // println!("diff: "{norm_yaw(norm_yaw(lby + data.yaw_additive) - eye_yaw):.2});
                            // println!("correct: "{norm_yaw(lby - eye_yaw):.2});
                        } else {
                            data.yaw_additive += distort.rate * (1.1 * 64.0);
                            data.yaw_additive -= distort.rate
                                * (ticks_since_distort_motion as f32 - distort.delta_ticks);
                            //
                            // println!("diff: "{norm_yaw(norm_yaw(lby + data.yaw_additive) - eye_yaw):.2});
                            // println!("correct: "{norm_yaw(lby - eye_yaw):.2});
                        }
                    }

                    // dbg!(data.lby_rate, distort.rate, rotation.rate, lby_base_offset);

                    // println!("diff: "{norm_yaw(norm_yaw(lby + data.yaw_additive) - eye_yaw):.2});
                    // println!("correct: "{norm_yaw(lby - eye_yaw):.2});
                }

                // for i in 0..(ticks_since_distort_motion as isize - distort.delta_ticks as isize)
                //     .max(0) as usize
                // {
                //     let forward_cycle = (i as usize / rotation.delta_ticks as usize) % 2 == 0;

                //     if forward_cycle {
                //         data.yaw_additive -= rotation
                //             .rate
                //             .abs()
                //             // .min(distort.rate.abs())
                //             .copysign(distort.rate);
                //     } else {
                //         data.yaw_additive += rotation
                //             .rate
                //             .abs()
                //             // .min(distort.rate.abs())
                //             .copysign(distort.rate);
                //     }
                // }
            }
            // (Some(distort), None) => {
            //     data.yaw_additive = if distort.rate > 0.0 { 120.0 } else { -120.0 };
            //     data.yaw_additive += distort.rate * distort.delta_ticks;

            //     for i in 0..ticks_since_distort_motion.max(distort.delta_ticks as usize) {
            //         let cycles = (i / distort.delta_ticks as usize) % 2 == 0;

            //         if i < distort.delta_ticks as usize || cycles {
            //             data.yaw_additive += distort.rate;
            //         } else {
            //             data.yaw_additive -= distort.rate;
            //         }
            //     }

            //     let inputs = [
            //         distort.rate * distort.delta_ticks,
            //         0.0,
            //         data.lby_rate.unwrap_or(0.0) * 64.0,
            //         (rotate_time - 1.1) / 1.1,
            //         ((ticks_since_distort_motion as f32 / distort.delta_ticks)
            //             * (core::f32::consts::PI / 2.0))
            //             .sin(),
            //         0.0,
            //     ];

            //     let predicted = if RESOLVER.is_none() {
            //         RESOLVER = Some(BaseResolver::new());
            //         [0.0; 2]
            //     } else {
            //         RESOLVER.as_ref().unwrap().forward(inputs)
            //     };
            //     let predicted_offset = predicted[0].atan2(predicted[1]).to_degrees();
            //     let predicted_yaw = norm_yaw(lby + predicted_offset);
            //     let delta_target_yaw = norm_yaw(predicted_yaw - eye_yaw);
            //     if delta_target_yaw.abs() > 35.0 / 2.0 {
            //         RESOLVER.as_mut().unwrap().train_step(inputs, target, 0.01);
            //         println!("diff: "{delta_target_yaw:.6});
            //     } else {
            //         println!("!");
            //     }
            // }
            _ => {
                data.yaw_additive = 0.0;
            }
        }
    }

    let resolved_yaw = norm_yaw(lby + data.yaw_additive);
    // FOR TESTING ONLY!!!
    let diff = norm_yaw(eye_yaw - resolved_yaw);
    let diff_lby = norm_yaw(eye_yaw - lby);

    if lby_accurate {
        println!("LBY!");
    }

    // println!("diff_lby: "{diff_lby:.2});

    // record previous values for next iterations calculations
    data.old_lby = Some(lby);
    data.old_yaw = Some(eye_yaw);
    data.old_animlayers = Some(*animlayers);

    // println!("0x"{global_vars as *const GlobalVarsBase as usize:X}", 0x"{animstate:X});
    return;
}

#[naked]
pub unsafe fn post_animstate_update() {
    naked_asm!(
        "pushad",
        "pushfd",
        "mov edx, edi",
        "call {animstate_update}",
        "popfd",
        "popad",
        "ret",
        animstate_update = sym animstate_update,
    );
}

pub fn init<'a, 'b>() -> Result<'a, Patch<'a, 'b, 6>>
where
    'b: 'a,
    'a: 'b,
{
    let server = Module::from_name(c"server.dll")?;

    println!("patch: 0x"{server.memory.as_ptr() as usize + ANIMSTATE_UPDATE_END_OFFSET:X});

    const ANIMSTATE_UPDATE_END_OFFSET: usize = 0x43346C;
    let patch = server.patch(ANIMSTATE_UPDATE_END_OFFSET, post_animstate_update as _)?;

    patch.enable();

    Ok(patch)
}

State result;
result.CON_pos_pos_x() = std::clamp(value.CON_pos_pos_x(),
                                    lower.CON_pos_pos_x(),
                                    upper.CON_pos_pos_x());
result.CON_pos_pos_y() = std::clamp(value.CON_pos_pos_y(),
                                    lower.CON_pos_pos_y(),
                                    upper.CON_pos_pos_y());
result.CON_pos_pos_z() = std::clamp(value.CON_pos_pos_z(),
                                    lower.CON_pos_pos_z(),
                                    upper.CON_pos_pos_z());
return result;

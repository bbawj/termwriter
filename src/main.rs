use std::collections::HashMap;
use std::fmt::Display;
use std::io::stdout;
use std::io::BufRead;
use std::io::BufReader;
use std::io::Write;
use std::process::Child;
use std::process::Command;
use std::process::Stdio;
use std::sync::mpsc;
use std::sync::mpsc::Receiver;
use std::thread;
use std::time::Duration;

use crossterm::cursor;
use crossterm::event;
use crossterm::event::PopKeyboardEnhancementFlags;
use crossterm::event::PushKeyboardEnhancementFlags;
use crossterm::style::Print;
use crossterm::style::PrintStyledContent;
use crossterm::style::Stylize;
use crossterm::terminal::disable_raw_mode;
use crossterm::terminal::enable_raw_mode;
use crossterm::terminal::Clear;
use crossterm::{terminal, QueueableCommand};
use lazy_static::lazy_static;

#[derive(Clone, Copy)]
enum ArmType {
    T0,
    T1,
    T2,
    T3,
}

#[derive(Clone, Copy)]
enum State {
    Rest,
    MoveUp,
    Strike,
    MoveDown,
}

#[derive(Clone, Copy)]
struct Arm {
    arm_type: ArmType,
    state: State,
}

impl Default for Arm {
    fn default() -> Self {
        Arm {
            arm_type: ArmType::T0,
            state: State::Rest,
        }
    }
}

#[derive(Clone)]
struct Draw {
    c: char,
    row_offset: i16,
    col_offset: i16,
}

type Canvas = Vec<Vec<char>>;

struct Typewriter {
    win_height: u16,
    height: u16,
    width: u16,
    paper_width: u16,
    mid: u16,
    pub canvas: Canvas,
    pub arms: Vec<Arm>,

    buffer: String,

    typewriter_upper: Vec<char>,
    typewriter_upper_slants: Vec<char>,
    typewriter_upper_bottom: Vec<char>,

    is_shift_pressed: bool,
    key_start: usize,
    prev_key: Option<&'static Key>,
}

impl Display for Typewriter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for row in &self.canvas[1..] {
            for c in row {
                write!(f, "{}", c)?;
            }
            write!(f, "{}", "\r\n")?;
        }
        Ok(())
    }
}

const TYPEWRITER_HEIGHT: u16 = 11;

const UPPER_ROW: usize = 1;
const UPPER_SLANTS_ROW: usize = 2;
const UPPER_BOTTOM_ROW: usize = 3;
const TYPEARM_ROW: usize = 4;
const TYPEARM_ROW_2: usize = 5;
const TYPEARM_ROW_3: usize = 6;
const KEY_ROW_1: usize = TYPEARM_ROW_3 + 1;

const N_KEY_ROWS: usize = 4;
const N_KEYS: usize = 46;
const KEYS_PER_ROW: usize = N_KEYS / N_KEY_ROWS;

const HORZ_UP: char = '⎺';
const HORZ: char = '─';
const VERT: char = '│';
const SLANTR_1: char = '/';
const SLANTR_2: char = '⟋';
const SLANTR_3: &str = "⎽⎼⎻⎺";
const SLANTR_3S: &str = "⎼⎻⎺";
const SLANTL_1: char = '\\';
const SLANTL_2: char = '⟍';
const SLANTL_3: &str = "⎺⎻⎼⎽";
const SLANTL_3S: &str = "⎺⎻⎼";
const KEY_1: &str = "- - - - - - - - - - - -";
const KEY_2: &str = " - - - - - - - - - - -";
const SPACE_BAR: [char; 23] = [HORZ; KEY_1.len()];

#[derive(Debug)]
struct Key {
    row: usize,
    col: usize,
    idx_in_row: usize,
}

const SHIFT: char = '\u{1}';
#[rustfmt::skip]
const QWERTY: [char; N_KEYS] = [
    '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '-', '=',
    'q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p','\u{8}',
    'a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l', ';', '\'', '\n',
    SHIFT, 'z', 'x', 'c', 'v', 'b', 'n', 'm', ',', '.', '/',
];

lazy_static! {
    static ref KEYMAPS: HashMap<char, Key> = HashMap::from_iter(
        QWERTY
            .into_iter()
            .scan((KEY_ROW_1, 0), |(cur_row, idx), c| {
                match c {
                    'q' | 'a' | SHIFT => {
                        *cur_row += 1;
                        *idx = 0;
                        Some((*cur_row, *idx, c))
                    }
                    _ => {
                        *idx += 1;
                        Some((*cur_row, *idx, c))
                    }
                }
            })
            // Uppercase
            .flat_map(|it| {
                match it.2 {
                    '1' => std::iter::once(it).chain(Some((it.0, it.1, '!'))),
                    '2' => std::iter::once(it).chain(Some((it.0, it.1, '@'))),
                    '3' => std::iter::once(it).chain(Some((it.0, it.1, '#'))),
                    '4' => std::iter::once(it).chain(Some((it.0, it.1, '$'))),
                    '5' => std::iter::once(it).chain(Some((it.0, it.1, '%'))),
                    '6' => std::iter::once(it).chain(Some((it.0, it.1, '^'))),
                    '7' => std::iter::once(it).chain(Some((it.0, it.1, '&'))),
                    '8' => std::iter::once(it).chain(Some((it.0, it.1, '*'))),
                    '9' => std::iter::once(it).chain(Some((it.0, it.1, '('))),
                    '0' => std::iter::once(it).chain(Some((it.0, it.1, ')'))),
                    '-' => std::iter::once(it).chain(Some((it.0, it.1, '_'))),
                    '=' => std::iter::once(it).chain(Some((it.0, it.1, '+'))),
                    ';' => std::iter::once(it).chain(Some((it.0, it.1, ':'))),
                    '\'' => std::iter::once(it).chain(Some((it.0, it.1, '"'))),
                    '/' => std::iter::once(it).chain(Some((it.0, it.1, '?'))),
                    _ => std::iter::once(it).chain(None),
                }
            })
            .map(|(row, idx, ch)| (
                ch,
                Key {
                    row,
                    col: idx * 2 + (row - KEY_ROW_1) % 2,
                    idx_in_row: idx,
                }
            ))
    );
}

impl Typewriter {
    pub fn new(height: u16, window_width: u16, win_height: u16) -> Box<Typewriter> {
        const TYPEWRITER_MIN_WIDTH: u16 = KEY_1.len() as u16;
        const MAX_PAPER_WIDTH: u16 = 90;
        // the smallest paper we can fit is the amount of characters we can type without moving the
        // paper off screen
        const MIN_PAPER_WIDTH: u16 = TYPEWRITER_MIN_WIDTH / 2;
        let paper_width = (window_width / 2).clamp(MIN_PAPER_WIDTH, MAX_PAPER_WIDTH);
        let width = (window_width - paper_width).min(paper_width) * 2;
        // dbg!(window_width, paper_width, width);

        let mut typewriter_upper = vec![' '; width.into()];
        let upper_space = 4;
        let typewriter_upper_width = width - upper_space * 2;
        for i in 1..typewriter_upper_width {
            typewriter_upper[(upper_space + i) as usize] = HORZ;
        }

        let mut typewriter_upper_slants = vec![' '; width.into()];
        SLANTR_3
            .chars()
            .enumerate()
            .for_each(|(i, c)| typewriter_upper_slants[i] = c);
        SLANTR_3
            .chars()
            .enumerate()
            .for_each(|(i, c)| typewriter_upper_slants[width as usize - 1 - i] = c);

        let mut typewriter_upper_bottom = vec![HORZ; width.into()];
        let upper_bottom_space = 2;
        SLANTL_3S
            .chars()
            .enumerate()
            .for_each(|(i, c)| typewriter_upper_bottom[upper_bottom_space + i] = c);
        SLANTL_3S.chars().enumerate().for_each(|(i, c)| {
            typewriter_upper_bottom[width as usize - upper_bottom_space - 1 - i] = c
        });

        let mut canvas = vec![vec![' '; width.into()]; height.into()];

        let mid = width / 2;
        let n_arms = (width / 2).min(N_KEYS as u16);
        let mid_arm = n_arms / 2;
        let mut arms = vec![Arm::default(); n_arms.into()];
        for (i, arm) in arms.iter_mut().enumerate() {
            arm.arm_type = if i as u16 == mid_arm {
                ArmType::T0
            } else if i < usize::from(mid_arm) / 2 || i > usize::from(mid_arm + mid_arm / 2) {
                ArmType::T2
            } else {
                ArmType::T1
            };
        }
        let spacing: usize = ((width - n_arms) / 2).into();
        // TODO: maybe a helper fn here would be nice
        SLANTL_3
            .chars()
            .enumerate()
            .for_each(|(i, c)| canvas[TYPEARM_ROW][spacing - i - 1] = c);
        SLANTR_3
            .chars()
            .enumerate()
            .for_each(|(i, c)| canvas[TYPEARM_ROW_2][spacing - i - 1] = c);
        SLANTL_3
            .chars()
            .enumerate()
            .for_each(|(i, c)| canvas[TYPEARM_ROW][width as usize - spacing + 1 + i] = c);
        SLANTR_3S
            .chars()
            .enumerate()
            .for_each(|(i, c)| canvas[TYPEARM_ROW_2][width as usize - spacing + 2 + i] = c);

        for i in spacing..spacing + n_arms as usize {
            canvas[TYPEARM_ROW_3][i] = HORZ;
        }

        let key_start = (width as usize - KEY_1.chars().count()) / 2;
        KEY_1
            .chars()
            .enumerate()
            .for_each(|(i, c)| canvas[KEY_ROW_1][key_start + i] = c);
        KEY_2
            .chars()
            .enumerate()
            .for_each(|(i, c)| canvas[KEY_ROW_1 + 1][key_start + i] = c);
        KEY_1
            .chars()
            .enumerate()
            .for_each(|(i, c)| canvas[KEY_ROW_1 + 2][key_start + i] = c);
        KEY_2
            .chars()
            .enumerate()
            .for_each(|(i, c)| canvas[KEY_ROW_1 + 3][key_start + i] = c);

        // let space_bar_down =
        // canvas[KEY_ROW_1 + 4].iter_mut().enumerate().for_each(|cell| )

        return Box::new(Typewriter {
            win_height,
            height,
            width,
            paper_width,
            canvas,
            arms,
            mid,
            typewriter_upper,
            typewriter_upper_slants,
            typewriter_upper_bottom,
            buffer: String::new(),
            key_start,
            prev_key: None,
            is_shift_pressed: false,
        });
    }

    pub fn update_state(&mut self, key: char, is_shift_pressed: bool) {
        match key.to_ascii_lowercase() {
            SHIFT => {
                assert!(false);
                let k = KEYMAPS.get(&key).unwrap();
                self.canvas[k.row][self.key_start + k.col] = '_';
                self.prev_key = Some(k);
            }
            ' ' => {
                if self.buffer.len() != self.paper_width.into() {
                    self.buffer.push(key);
                }
            }
            _ => {
                if let Some(k) = KEYMAPS.get(&key.to_ascii_lowercase()) {
                    self.is_shift_pressed = is_shift_pressed;

                    let pos = self.get_key_pos(k);
                    if self.buffer.len() != self.paper_width.into() {
                        self.buffer.push(key);
                    }
                    self.arms[pos].state = State::MoveUp;
                    if let Some(p) = self.prev_key {
                        self.canvas[p.row][self.key_start + p.col] = '-';
                    }
                    self.canvas[k.row][self.key_start + k.col] = '_';
                    self.prev_key = Some(k);
                }
            }
        }
    }

    pub fn refresh(&mut self) -> Result<(), anyhow::Error> {
        self.canvas[UPPER_ROW].copy_from_slice(&self.typewriter_upper);
        self.canvas[UPPER_SLANTS_ROW].copy_from_slice(&self.typewriter_upper_slants);
        self.canvas[UPPER_BOTTOM_ROW].copy_from_slice(&self.typewriter_upper_bottom);
        self.update_arms();
        let mut stdout = stdout();
        self.draw_text(&self.buffer)?;
        stdout.queue(Clear(terminal::ClearType::FromCursorDown))?;
        stdout.queue(Print(&self))?;
        stdout.queue(cursor::MoveUp(self.height))?;
        stdout.flush()?;
        Ok(())
    }

    fn draw_text(&self, text: &str) -> Result<(), anyhow::Error> {
        let mut stdout = stdout();
        let print_pos = self.mid as usize - self.buffer.len();
        // TODO: how to draw the striking cursor?
        crossterm::queue!(
            stdout,
            cursor::Hide,
            Print(" ".repeat(print_pos)),
            PrintStyledContent(text.black().on_white()),
        )?;
        let (col, _) = cursor::position()?;
        crossterm::queue!(
            stdout,
            PrintStyledContent(
                " ".repeat(
                    (print_pos + self.paper_width as usize)
                        .checked_sub(col.into())
                        .unwrap_or(0)
                )
                .on_white()
            ),
            terminal::Clear(terminal::ClearType::UntilNewLine),
            Print("\r\n"),
        )?;
        Ok(())
    }

    fn get_key_pos(&self, key: &Key) -> usize {
        // let spacing = self.width as usize / 2 - KEYS_PER_ROW;
        let mid_arm = self.arms.len() / 2;
        let lerp = self.arms.len() as f32 / N_KEYS as f32;
        let offset = (lerp
            * (key.idx_in_row.abs_diff(KEYS_PER_ROW / 2)) as f32
            * (key.row - KEY_ROW_1 + 1) as f32)
            .round() as usize;
        // dbg!(mid_arm, lerp, key, offset);
        let pos = if key.col + self.key_start > self.mid.into() {
            mid_arm + offset
        } else {
            mid_arm - offset
        };
        return pos;
    }

    pub fn update_arms(&mut self) {
        let mid_arm = self.arms.len() / 2;
        let spacing = (self.width - self.arms.len() as u16) / 2;
        let shifted: i16 = if self.is_shift_pressed { -1 } else { 0 };
        for (i, arm) in self.arms.iter_mut().enumerate() {
            let flip = if i < mid_arm { 0 } else { 1 };
            let cmds = Self::draw_arm(&arm, flip != 0, i as u16, mid_arm as u16);
            for cmd in cmds {
                self.canvas[usize::try_from(TYPEARM_ROW as i16 + shifted + cmd.row_offset)
                    .unwrap_or_default()]
                    [usize::try_from((spacing as i16 + i as i16) + cmd.col_offset).unwrap()] =
                    cmd.c;
            }
            arm.state = match arm.state {
                State::Rest => State::Rest,
                State::MoveUp => State::Strike,
                State::Strike => State::MoveDown,
                State::MoveDown => State::Rest,
            }
        }
    }

    fn draw_arm(arm: &Arm, flip: bool, start: u16, mid: u16) -> Vec<Draw> {
        let diff = mid.abs_diff(start);
        match arm.arm_type {
            ArmType::T0 => match arm.state {
                State::Rest => vec![
                    Draw {
                        c: VERT,
                        row_offset: 0,
                        col_offset: 0,
                    },
                    Draw {
                        c: VERT,
                        row_offset: 1,
                        col_offset: 0,
                    },
                ],
                State::MoveUp | State::MoveDown => vec![
                    Draw {
                        c: ' ',
                        row_offset: 1,
                        col_offset: 0,
                    },
                    Draw {
                        c: VERT,
                        row_offset: 0,
                        col_offset: 0,
                    },
                    Draw {
                        c: '_',
                        row_offset: -1,
                        col_offset: 0,
                    },
                ],
                State::Strike => Self::generate_strike_arm(arm.arm_type, flip, diff),
            },
            ArmType::T1 => match arm.state {
                State::Rest => vec![
                    Draw {
                        c: if flip { SLANTL_1 } else { SLANTR_1 },
                        row_offset: 0,
                        col_offset: 0,
                    },
                    Draw {
                        c: if flip { SLANTL_1 } else { SLANTR_1 },
                        row_offset: 1,
                        col_offset: if flip { 1 } else { -1 },
                    },
                ],
                State::MoveUp | State::MoveDown => vec![
                    Draw {
                        c: ' ',
                        row_offset: 1,
                        col_offset: if flip { 1 } else { -1 },
                    },
                    Draw {
                        c: if flip { SLANTR_2 } else { SLANTL_2 },
                        row_offset: 0,
                        col_offset: 0,
                    },
                    Draw {
                        c: if flip { SLANTR_2 } else { SLANTL_2 },
                        row_offset: -1,
                        col_offset: if flip { 1 } else { -1 },
                    },
                ],
                State::Strike => Self::generate_strike_arm(arm.arm_type, flip, diff),
            },
            ArmType::T2 => match arm.state {
                State::Rest => vec![
                    Draw {
                        c: if flip { SLANTL_2 } else { SLANTR_2 },
                        row_offset: 0,
                        col_offset: 0,
                    },
                    Draw {
                        c: if flip { SLANTL_2 } else { SLANTR_2 },
                        row_offset: 1,
                        col_offset: if flip { 1 } else { -1 },
                    },
                ],
                State::MoveUp | State::MoveDown => vec![
                    Draw {
                        c: ' ',
                        row_offset: 1,
                        col_offset: if flip { 1 } else { -1 },
                    },
                    Draw {
                        c: if flip { SLANTR_2 } else { SLANTL_2 },
                        row_offset: 0,
                        col_offset: 0,
                    },
                    Draw {
                        c: if flip { SLANTR_2 } else { SLANTL_2 },
                        row_offset: -1,
                        col_offset: if flip { 1 } else { -1 },
                    },
                ],
                State::Strike => Self::generate_strike_arm(arm.arm_type, flip, diff),
            },
            ArmType::T3 => todo!(),
        }
    }

    fn generate_strike_arm(arm_type: ArmType, flip: bool, horz_dist: u16) -> Vec<Draw> {
        let mut cmds = vec![Draw {
            c: VERT,
            row_offset: -1,
            col_offset: 0,
        }];
        let vert_dist = 2;
        match arm_type {
            ArmType::T0 => {
                for i in 0..vert_dist {
                    cmds.push(Draw {
                        c: VERT,
                        row_offset: -(2 + i),
                        col_offset: 0,
                    })
                }
            }
            ArmType::T1 => {
                for i in 0..vert_dist {
                    cmds.push(Draw {
                        c: if flip { SLANTL_1 } else { SLANTR_1 },
                        row_offset: -(2 + i),
                        col_offset: if flip { -i } else { i },
                    });
                }
                for i in 2 as i16..horz_dist.try_into().unwrap() {
                    cmds.push(Draw {
                        c: HORZ_UP,
                        row_offset: -3,
                        col_offset: if flip { -i } else { i },
                    })
                }
            }
            ArmType::T2 => {
                for i in 0..vert_dist {
                    cmds.push(Draw {
                        c: if flip { SLANTL_2 } else { SLANTR_2 },
                        row_offset: -(2 + i),
                        col_offset: if flip { -(i * 2 + 1) } else { i * 2 + 1 },
                    });
                }
                for i in 5 as i16..horz_dist.try_into().unwrap() {
                    cmds.push(Draw {
                        c: HORZ_UP,
                        row_offset: -3,
                        col_offset: if flip { -i } else { i },
                    })
                }
            }
            ArmType::T3 => todo!(),
        };
        cmds.push(Draw {
            c: VERT,
            row_offset: -4,
            col_offset: if flip {
                -(horz_dist as i16)
            } else {
                horz_dist as i16
            },
        });
        return cmds;
    }
}

fn main() -> Result<(), anyhow::Error> {
    let window_size = terminal::window_size()?;

    let mut typewriter = Typewriter::new(TYPEWRITER_HEIGHT, 60, window_size.rows);

    let mut stdout = stdout();
    stdout.queue(PushKeyboardEnhancementFlags(
        event::KeyboardEnhancementFlags::DISAMBIGUATE_ESCAPE_CODES,
    ))?;
    stdout.queue(PushKeyboardEnhancementFlags(
        event::KeyboardEnhancementFlags::REPORT_ALL_KEYS_AS_ESCAPE_CODES,
    ))?;
    stdout.queue(cursor::Hide)?;
    typewriter.refresh()?;

    let mut child = Command::new("sh")
        .stdout(Stdio::piped())
        .stdin(Stdio::piped())
        .spawn()?;

    // Read and parse output from the pty with reader
    // let reader = BufReader::new(pair.master.try_clone_reader()?);
    let reader = BufReader::new(child.stdout.take().unwrap());

    // Send data to the pty by writing to the master
    // let mut writer = pair.master.take_writer()?;
    let mut writer = child.stdin.take().unwrap();
    let rx = spawn_pty_channel(reader);
    // let mut written = false;
    enable_raw_mode()?;
    loop {
        // TODO: use delta to cap FPS
        // TODO: print FPS
        if event::poll(Duration::from_millis(33))? {
            match event::read()? {
                event::Event::Key(e) => match e.code {
                    event::KeyCode::Backspace => {
                        typewriter.buffer.pop();
                    }
                    event::KeyCode::Enter => {
                        writeln!(writer, "{}", typewriter.buffer)?;
                        // dbg!("test");
                        writer.flush()?;
                        // written = true;
                    }
                    event::KeyCode::Tab => todo!(),
                    event::KeyCode::Char(c) => {
                        if c == 'c' && e.modifiers.contains(event::KeyModifiers::CONTROL) {
                            break;
                        }
                        typewriter
                            .update_state(c, e.modifiers.contains(event::KeyModifiers::SHIFT));
                    }
                    event::KeyCode::Esc => todo!(),
                    event::KeyCode::CapsLock => todo!(),
                    event::KeyCode::Modifier(m) => {
                        if m == event::ModifierKeyCode::LeftShift {
                            typewriter.update_state(SHIFT, false);
                        }
                    }
                    _ => continue,
                },
                _ => todo!(),
            };
        } else {
            if let Some(p) = typewriter.prev_key {
                typewriter.canvas[p.row][typewriter.key_start + p.col] = '-';
            }
        }
        match rx.try_recv() {
            Ok(s) => {
                // dbg!(&s);
                s.as_bytes()
                    .chunks(typewriter.paper_width.into())
                    .for_each(|c| {
                        typewriter
                            .draw_text(std::str::from_utf8(c).unwrap())
                            .unwrap()
                    });
            }
            Err(mpsc::TryRecvError::Empty) => {
                // stdout.write(b"\n")?;
                // writeln!(stdout, "{}\n", s.as_bytes())?;
            }
            Err(mpsc::TryRecvError::Disconnected) => {
                break;
            }
        }
        typewriter.refresh()?;
    }
    cleanup(child);
    Ok(())
}

fn cleanup(mut child: Child) {
    let _ = child.kill();
    let _ = disable_raw_mode();
    let _ = crossterm::execute!(stdout(), PopKeyboardEnhancementFlags, cursor::Show);
}

fn spawn_pty_channel(mut reader: impl BufRead + std::marker::Send + 'static) -> Receiver<String> {
    let (tx, rx) = mpsc::channel::<String>();
    thread::spawn(move || loop {
        let mut buffer = String::new();
        if let Err(_) = reader.read_line(&mut buffer) {
            break;
        };
        if let Err(_) = tx.send(strip_ansi(&buffer)) {
            break;
        }
    });
    rx
}

fn strip_ansi(text: &str) -> String {
    // let re = Regex::new(r"\x1b\[[0-9]*[ -/]*m").unwrap();
    // let sanitized = re.replace_all(text, "");
    text.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_pos() {
        let t = Typewriter::new(TYPEWRITER_HEIGHT, 180, 180);
        assert_eq!(46, t.arms.len());
        // assert_eq!(10, t.get_key_pos(KEYMAPS.get(&'1').unwrap()));
        // assert_eq!(15, t.get_key_pos(KEYMAPS.get(&'0').unwrap()));
        assert_eq!(0, t.get_key_pos(KEYMAPS.get(&'z').unwrap()));
    }
}

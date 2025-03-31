#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2024.2.4),
    on März 27, 2025, at 15:46
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
prefs.hardware['audioLatencyMode'] = '3'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
from psychopy.tools import environmenttools
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER, priority)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

import psychopy.iohub as io
from psychopy.hardware import keyboard

# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2024.2.4'
expName = 'masculine overcompensation experiment - willer et al. 2013'  # from the Builder filename that created this script
# information about this experiment
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'session': '001',
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'psychopyVersion|hid': psychopyVersion,
}

# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = True
_winSize = [2560, 1440]
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version='',
        extraInfo=expInfo, runtimeInfo=None,
        originPath='C:\\Users\\Simon\\Desktop\\Praktikum\\1. Projekt Experiment von Willer et. al nachbauen\\psychopy\\masculine overcompensation experiment - willer et al. 2013_lastrun.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # set how much information should be printed to the console / app
    if PILOTING:
        logging.console.setLevel(
            prefs.piloting['pilotConsoleLoggingLevel']
        )
    else:
        logging.console.setLevel('warning')
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log')
    if PILOTING:
        logFile.setLevel(
            prefs.piloting['pilotLoggingLevel']
        )
    else:
        logFile.setLevel(
            logging.getLevel('info')
        )
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if PILOTING:
        logging.debug('Fullscreen settings ignored as running in pilot mode.')
    
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=0,
            winType='pyglet', allowGUI=True, allowStencil=False,
            monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height',
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [0,0,0]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    win.hideMessage()
    # show a visual indicator if we're in piloting mode
    if PILOTING and prefs.piloting['showPilotingIndicator']:
        win.showPilotingIndicator()
    
    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to 
    the device manager (deviceManager)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}
    
    # Setup iohub keyboard
    ioConfig['Keyboard'] = dict(use_keymap='psychopy')
    
    # Setup iohub experiment
    ioConfig['Experiment'] = dict(filename=thisExp.dataFileName)
    
    # Start ioHub server
    ioServer = io.launchHubServer(window=win, **ioConfig)
    
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='iohub'
        )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], playbackComponents=[]):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    playbackComponents : list, tuple
        List of any components with a `pause` method which need to be paused.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # start a timer to figure out how long we're paused for
    pauseTimer = core.Clock()
    # pause any playback components
    for comp in playbackComponents:
        comp.pause()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='ioHub',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # sleep 1ms so other threads can execute
        clock.time.sleep(0.001)
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    for comp in playbackComponents:
        comp.play()
    # reset any timers
    for timer in timers:
        timer.addTime(-pauseTimer.getTime())


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure window is set to foreground to prevent losing focus
    win.winHandle.activate()
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ioHub'
        )
    eyetracker = deviceManager.getDevice('eyetracker')
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "Welcome" ---
    Welcometext = visual.TextStim(win=win, name='Welcometext',
        text='Dear participant,\nThank you for taking part in our survey on the topic of gender identity.\nFollowing you will find some questions about you as a person and your perspective on different statements mainly regarded to your identity.\n\nWe ask you to select the appropriate answers when completing the questionnaire. Please note that we are interested in your personal impression. There are no right or wrong answers. The survey takes about 10 minutes. \n\n\n',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    button = visual.ButtonStim(win, 
        text='Click here to continue', font='Open Sans',
        pos=(0, -0.2),
        letterHeight=0.03,
        size=(0.3, 0.1), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor='darkgrey', borderColor=None,
        color='white', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='button',
        depth=-1
    )
    button.buttonClock = core.Clock()
    # Run 'Begin Experiment' code from code
    import random
    chooseExperiment = random.randint(0, 1)
    
    # --- Initialize components for Routine "Welcome_2" ---
    Welcometext_2 = visual.TextStim(win=win, name='Welcometext_2',
        text='Participation in the survey is voluntary. You can revoke your consent to participate in this study at any time and without giving reasons, without incurring any disadvantages. To do so, simply press "Esc". \n\nAll information provided by study participants is treated as strictly confidential and anonymized. The data collected will be used exclusively for scientific purposes.\n\n\n',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    button_2 = visual.ButtonStim(win, 
        text='Click here to continue', font='Open Sans',
        pos=(0, -0.2),
        letterHeight=0.03,
        size=(0.3, 0.1), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor='darkgrey', borderColor=None,
        color='white', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='button_2',
        depth=-1
    )
    button_2.buttonClock = core.Clock()
    
    # --- Initialize components for Routine "demographics" ---
    slider_age = visual.Slider(win=win, name='slider_age',
        startValue=None, size=(1.0, 0.1), pos=(0, 0), units=win.units,
        labels=['18-24','25-34','35-44','45-54','55-64','65-99'], ticks=(1, 2, 3, 4, 5, 6), granularity=1.0,
        style='rating', styleTweaks=(), opacity=None,
        labelColor='LightGray', markerColor='Red', lineColor='White', colorSpace='rgb',
        font='Open Sans', labelHeight=0.02,
        flip=False, ori=0.0, depth=0, readOnly=False)
    textage = visual.TextStim(win=win, name='textage',
        text='What is your age?',
        font='Open Sans',
        pos=(0, 0.25), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "pretext_gender_identity" ---
    gender_identity_pretext_2 = visual.TextStim(win=win, name='gender_identity_pretext_2',
        text='Now we will start with the gender identity survey with which we will calcualte your gender identity on a dimension of masculine to feminine.\n\nFor each of the following words, please rate on the scale how well you think the word describes yourself:',
        font='open Sans',
        pos=(0, 0), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    button_3 = visual.ButtonStim(win, 
        text='Click here to continue', font='Open Sans',
        pos=(0, -0.2),
        letterHeight=0.03,
        size=(0.3, 0.1), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor='darkgrey', borderColor=None,
        color='white', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='button_3',
        depth=-1
    )
    button_3.buttonClock = core.Clock()
    
    # --- Initialize components for Routine "blank_wait" ---
    waittext = visual.TextStim(win=win, name='waittext',
        text='Please wait a few seconds while we will calculate your gender identity score. It will be shown to you on the next page...',
        font='Open Sans',
        pos=(0, 0.1), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "feedback_score_masculine" ---
    image = visual.ImageStim(
        win=win,
        name='image', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, -0.09), draggable=False, size=(1.1, .25),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    button_4 = visual.ButtonStim(win, 
        text='Click here to continue', font='Open Sans',
        pos=(0, -0.35),
        letterHeight=0.03,
        size=(0.3, 0.1), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor='darkgrey', borderColor=None,
        color='white', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='button_4',
        depth=-1
    )
    button_4.buttonClock = core.Clock()
    text_masculine = visual.TextStim(win=win, name='text_masculine',
        text='The following is your score on the gender identity survey. It has been placed on a 0 to 50 index running from "Masculine" to "Feminine". Those lower on the scale have more masculine gender identities, those higher on the scale have more feminine gender identities.\n\nYour Score: 11\n\nBelow is a line graph of average score of men and women on the Gender Identity Survey. We have indicated your score with an "X" on the line.\n\n',
        font='Open Sans',
        pos=(0, .25), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "feedback_score_feminine" ---
    button_9 = visual.ButtonStim(win, 
        text='Click here to continue', font='Open Sans',
        pos=(0, -0.35),
        letterHeight=0.03,
        size=(0.3, 0.1), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor='darkgrey', borderColor=None,
        color='white', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='button_9',
        depth=0
    )
    button_9.buttonClock = core.Clock()
    text_feminine = visual.TextStim(win=win, name='text_feminine',
        text='The following is your score on the gender identity survey. It has been placed on a 0 to 50 index running from "Masculine" to "Feminine". Those lower on the scale have more masculine gender identities, those higher on the scale have more feminine gender identities.\n\nYour Score: 32\n\nBelow is a line graph of average score of men and women on the Gender Identity Survey. We have indicated your score with an "X" on the line.\n\n',
        font='Open Sans',
        pos=(0, .2), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    image_2 = visual.ImageStim(
        win=win,
        name='image_2', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, -0.09), draggable=False, size=(1.1, .25),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-2.0)
    
    # --- Initialize components for Routine "pretext_group_based_dom" ---
    pretext_group_based_dominance = visual.TextStim(win=win, name='pretext_group_based_dominance',
        text='Now we will ask you some more general questions about your personality. \n\nFor each of the following statements, please rate on the scale how well you think the statement describes yourself:',
        font='open Sans',
        pos=(0, 0), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    button_5 = visual.ButtonStim(win, 
        text='Click here to continue', font='Open Sans',
        pos=(0, -0.2),
        letterHeight=0.03,
        size=(0.3, 0.1), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor='darkgrey', borderColor=None,
        color='white', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='button_5',
        depth=-1
    )
    button_5.buttonClock = core.Clock()
    
    # --- Initialize components for Routine "trial_group_based_dominance" ---
    slider_3 = visual.Slider(win=win, name='slider_3',
        startValue=None, size=(1.0, 0.1), pos=(0, 0), units=win.units,
        labels=['Almost Never True', 'Almost Always True'], ticks=(1, 2, 3, 4, 5, 6, 7,8,9,10), granularity=1.0,
        style='rating', styleTweaks=(), opacity=None,
        labelColor='LightGray', markerColor='Red', lineColor='White', colorSpace='rgb',
        font='Open Sans', labelHeight=0.02,
        flip=False, ori=0.0, depth=0, readOnly=False)
    text5 = visual.TextStim(win=win, name='text5',
        text='',
        font='Open Sans',
        pos=(0, 0.25), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "pretext_political_conservatism" ---
    pretext_political_position = visual.TextStim(win=win, name='pretext_political_position',
        text="Finally we've got some political and societal statements.  \nPlease rate on the scale what you feel or think about them:",
        font='open Sans',
        pos=(0, 0), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    button_6 = visual.ButtonStim(win, 
        text='Click here to continue', font='Open Sans',
        pos=(0, -0.2),
        letterHeight=0.03,
        size=(0.3, 0.1), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor='darkgrey', borderColor=None,
        color='white', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='button_6',
        depth=-1
    )
    button_6.buttonClock = core.Clock()
    
    # --- Initialize components for Routine "thank_you" ---
    Thank_you = visual.TextStim(win=win, name='Thank_you',
        text='Thank you for taking the time to participate in our study.\n\nFirst of all, we need to tell you that we didn´t measure your gender identity score, you were randomly assigned to one of two scores. One groupe which had the "11" as a score, in the average area of masculine, and the other group had the "32" score, in the average area of feminine. We needed to pretend this in order to eventually create an effect of masulinity threat on the participants. Our main goal of this study was to investigate if there are an masculine oversompensation effect resulting from masculinity threat towards group-based dominance and political conservatism.\n\nWe would like to ask you not to discuss the exact aim of the study with people who may still participate, as this could influence the results.\n\nFor any concerns or questions regarding the study please contact the following:\nName: example name\nEmail: example@university\n',
        font='Open Sans',
        pos=(0, .1), draggable=False, height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    button_8 = visual.ButtonStim(win, 
        text='Click here to end the experiment', font='Open Sans',
        pos=(0, -0.33),
        letterHeight=0.03,
        size=(0.3, 0.1), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor='darkgrey', borderColor=None,
        color='white', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='button_8',
        depth=-1
    )
    button_8.buttonClock = core.Clock()
    
    # create some handy timers
    
    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # --- Prepare to start Routine "Welcome" ---
    # create an object to store info about Routine Welcome
    Welcome = data.Routine(
        name='Welcome',
        components=[Welcometext, button],
    )
    Welcome.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # reset button to account for continued clicks & clear times on/off
    button.reset()
    # store start times for Welcome
    Welcome.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    Welcome.tStart = globalClock.getTime(format='float')
    Welcome.status = STARTED
    thisExp.addData('Welcome.started', Welcome.tStart)
    Welcome.maxDuration = None
    # keep track of which components have finished
    WelcomeComponents = Welcome.components
    for thisComponent in Welcome.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Welcome" ---
    Welcome.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *Welcometext* updates
        
        # if Welcometext is starting this frame...
        if Welcometext.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            Welcometext.frameNStart = frameN  # exact frame index
            Welcometext.tStart = t  # local t and not account for scr refresh
            Welcometext.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Welcometext, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'Welcometext.started')
            # update status
            Welcometext.status = STARTED
            Welcometext.setAutoDraw(True)
        
        # if Welcometext is active this frame...
        if Welcometext.status == STARTED:
            # update params
            pass
        # *button* updates
        
        # if button is starting this frame...
        if button.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            button.frameNStart = frameN  # exact frame index
            button.tStart = t  # local t and not account for scr refresh
            button.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(button, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'button.started')
            # update status
            button.status = STARTED
            win.callOnFlip(button.buttonClock.reset)
            button.setAutoDraw(True)
        
        # if button is active this frame...
        if button.status == STARTED:
            # update params
            pass
            # check whether button has been pressed
            if button.isClicked:
                if not button.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    button.timesOn.append(button.buttonClock.getTime())
                    button.timesOff.append(button.buttonClock.getTime())
                elif len(button.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    button.timesOff[-1] = button.buttonClock.getTime()
                if not button.wasClicked:
                    # end routine when button is clicked
                    continueRoutine = False
                if not button.wasClicked:
                    # run callback code when button is clicked
                    pass
        # take note of whether button was clicked, so that next frame we know if clicks are new
        button.wasClicked = button.isClicked and button.status == STARTED
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            Welcome.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in Welcome.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Welcome" ---
    for thisComponent in Welcome.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for Welcome
    Welcome.tStop = globalClock.getTime(format='float')
    Welcome.tStopRefresh = tThisFlipGlobal
    thisExp.addData('Welcome.stopped', Welcome.tStop)
    thisExp.addData('button.numClicks', button.numClicks)
    if button.numClicks:
       thisExp.addData('button.timesOn', button.timesOn)
       thisExp.addData('button.timesOff', button.timesOff)
    else:
       thisExp.addData('button.timesOn', "")
       thisExp.addData('button.timesOff', "")
    thisExp.nextEntry()
    # the Routine "Welcome" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "Welcome_2" ---
    # create an object to store info about Routine Welcome_2
    Welcome_2 = data.Routine(
        name='Welcome_2',
        components=[Welcometext_2, button_2],
    )
    Welcome_2.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # reset button_2 to account for continued clicks & clear times on/off
    button_2.reset()
    # store start times for Welcome_2
    Welcome_2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    Welcome_2.tStart = globalClock.getTime(format='float')
    Welcome_2.status = STARTED
    thisExp.addData('Welcome_2.started', Welcome_2.tStart)
    Welcome_2.maxDuration = None
    # keep track of which components have finished
    Welcome_2Components = Welcome_2.components
    for thisComponent in Welcome_2.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Welcome_2" ---
    Welcome_2.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *Welcometext_2* updates
        
        # if Welcometext_2 is starting this frame...
        if Welcometext_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            Welcometext_2.frameNStart = frameN  # exact frame index
            Welcometext_2.tStart = t  # local t and not account for scr refresh
            Welcometext_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Welcometext_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'Welcometext_2.started')
            # update status
            Welcometext_2.status = STARTED
            Welcometext_2.setAutoDraw(True)
        
        # if Welcometext_2 is active this frame...
        if Welcometext_2.status == STARTED:
            # update params
            pass
        # *button_2* updates
        
        # if button_2 is starting this frame...
        if button_2.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            button_2.frameNStart = frameN  # exact frame index
            button_2.tStart = t  # local t and not account for scr refresh
            button_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(button_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'button_2.started')
            # update status
            button_2.status = STARTED
            win.callOnFlip(button_2.buttonClock.reset)
            button_2.setAutoDraw(True)
        
        # if button_2 is active this frame...
        if button_2.status == STARTED:
            # update params
            pass
            # check whether button_2 has been pressed
            if button_2.isClicked:
                if not button_2.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    button_2.timesOn.append(button_2.buttonClock.getTime())
                    button_2.timesOff.append(button_2.buttonClock.getTime())
                elif len(button_2.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    button_2.timesOff[-1] = button_2.buttonClock.getTime()
                if not button_2.wasClicked:
                    # end routine when button_2 is clicked
                    continueRoutine = False
                if not button_2.wasClicked:
                    # run callback code when button_2 is clicked
                    pass
        # take note of whether button_2 was clicked, so that next frame we know if clicks are new
        button_2.wasClicked = button_2.isClicked and button_2.status == STARTED
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            Welcome_2.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in Welcome_2.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Welcome_2" ---
    for thisComponent in Welcome_2.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for Welcome_2
    Welcome_2.tStop = globalClock.getTime(format='float')
    Welcome_2.tStopRefresh = tThisFlipGlobal
    thisExp.addData('Welcome_2.stopped', Welcome_2.tStop)
    thisExp.addData('button_2.numClicks', button_2.numClicks)
    if button_2.numClicks:
       thisExp.addData('button_2.timesOn', button_2.timesOn)
       thisExp.addData('button_2.timesOff', button_2.timesOff)
    else:
       thisExp.addData('button_2.timesOn', "")
       thisExp.addData('button_2.timesOff', "")
    thisExp.nextEntry()
    # the Routine "Welcome_2" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "demographics" ---
    # create an object to store info about Routine demographics
    demographics = data.Routine(
        name='demographics',
        components=[slider_age, textage],
    )
    demographics.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    slider_age.reset()
    # store start times for demographics
    demographics.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    demographics.tStart = globalClock.getTime(format='float')
    demographics.status = STARTED
    thisExp.addData('demographics.started', demographics.tStart)
    demographics.maxDuration = None
    # keep track of which components have finished
    demographicsComponents = demographics.components
    for thisComponent in demographics.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "demographics" ---
    demographics.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *slider_age* updates
        
        # if slider_age is starting this frame...
        if slider_age.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            slider_age.frameNStart = frameN  # exact frame index
            slider_age.tStart = t  # local t and not account for scr refresh
            slider_age.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(slider_age, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'slider_age.started')
            # update status
            slider_age.status = STARTED
            slider_age.setAutoDraw(True)
        
        # if slider_age is active this frame...
        if slider_age.status == STARTED:
            # update params
            pass
        
        # Check slider_age for response to end Routine
        if slider_age.getRating() is not None and slider_age.status == STARTED:
            continueRoutine = False
        
        # *textage* updates
        
        # if textage is starting this frame...
        if textage.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            textage.frameNStart = frameN  # exact frame index
            textage.tStart = t  # local t and not account for scr refresh
            textage.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(textage, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'textage.started')
            # update status
            textage.status = STARTED
            textage.setAutoDraw(True)
        
        # if textage is active this frame...
        if textage.status == STARTED:
            # update params
            pass
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            demographics.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in demographics.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "demographics" ---
    for thisComponent in demographics.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for demographics
    demographics.tStop = globalClock.getTime(format='float')
    demographics.tStopRefresh = tThisFlipGlobal
    thisExp.addData('demographics.stopped', demographics.tStop)
    thisExp.addData('slider_age.response', slider_age.getRating())
    thisExp.addData('slider_age.rt', slider_age.getRT())
    thisExp.nextEntry()
    # the Routine "demographics" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "pretext_gender_identity" ---
    # create an object to store info about Routine pretext_gender_identity
    pretext_gender_identity = data.Routine(
        name='pretext_gender_identity',
        components=[gender_identity_pretext_2, button_3],
    )
    pretext_gender_identity.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # reset button_3 to account for continued clicks & clear times on/off
    button_3.reset()
    # store start times for pretext_gender_identity
    pretext_gender_identity.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    pretext_gender_identity.tStart = globalClock.getTime(format='float')
    pretext_gender_identity.status = STARTED
    thisExp.addData('pretext_gender_identity.started', pretext_gender_identity.tStart)
    pretext_gender_identity.maxDuration = None
    # keep track of which components have finished
    pretext_gender_identityComponents = pretext_gender_identity.components
    for thisComponent in pretext_gender_identity.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "pretext_gender_identity" ---
    pretext_gender_identity.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *gender_identity_pretext_2* updates
        
        # if gender_identity_pretext_2 is starting this frame...
        if gender_identity_pretext_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            gender_identity_pretext_2.frameNStart = frameN  # exact frame index
            gender_identity_pretext_2.tStart = t  # local t and not account for scr refresh
            gender_identity_pretext_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(gender_identity_pretext_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'gender_identity_pretext_2.started')
            # update status
            gender_identity_pretext_2.status = STARTED
            gender_identity_pretext_2.setAutoDraw(True)
        
        # if gender_identity_pretext_2 is active this frame...
        if gender_identity_pretext_2.status == STARTED:
            # update params
            pass
        # *button_3* updates
        
        # if button_3 is starting this frame...
        if button_3.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            button_3.frameNStart = frameN  # exact frame index
            button_3.tStart = t  # local t and not account for scr refresh
            button_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(button_3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'button_3.started')
            # update status
            button_3.status = STARTED
            win.callOnFlip(button_3.buttonClock.reset)
            button_3.setAutoDraw(True)
        
        # if button_3 is active this frame...
        if button_3.status == STARTED:
            # update params
            pass
            # check whether button_3 has been pressed
            if button_3.isClicked:
                if not button_3.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    button_3.timesOn.append(button_3.buttonClock.getTime())
                    button_3.timesOff.append(button_3.buttonClock.getTime())
                elif len(button_3.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    button_3.timesOff[-1] = button_3.buttonClock.getTime()
                if not button_3.wasClicked:
                    # end routine when button_3 is clicked
                    continueRoutine = False
                if not button_3.wasClicked:
                    # run callback code when button_3 is clicked
                    pass
        # take note of whether button_3 was clicked, so that next frame we know if clicks are new
        button_3.wasClicked = button_3.isClicked and button_3.status == STARTED
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            pretext_gender_identity.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in pretext_gender_identity.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "pretext_gender_identity" ---
    for thisComponent in pretext_gender_identity.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for pretext_gender_identity
    pretext_gender_identity.tStop = globalClock.getTime(format='float')
    pretext_gender_identity.tStopRefresh = tThisFlipGlobal
    thisExp.addData('pretext_gender_identity.stopped', pretext_gender_identity.tStop)
    thisExp.addData('button_3.numClicks', button_3.numClicks)
    if button_3.numClicks:
       thisExp.addData('button_3.timesOn', button_3.timesOn)
       thisExp.addData('button_3.timesOff', button_3.timesOff)
    else:
       thisExp.addData('button_3.timesOn', "")
       thisExp.addData('button_3.timesOff', "")
    thisExp.nextEntry()
    # the Routine "pretext_gender_identity" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    loop_gender_identity = data.TrialHandler2(
        name='loop_gender_identity',
        nReps=1.0, 
        method='sequential', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('items_gender_identiity_survey.xlsx'), 
        seed=None, 
    )
    thisExp.addLoop(loop_gender_identity)  # add the loop to the experiment
    thisLoop_gender_identity = loop_gender_identity.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisLoop_gender_identity.rgb)
    if thisLoop_gender_identity != None:
        for paramName in thisLoop_gender_identity:
            globals()[paramName] = thisLoop_gender_identity[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisLoop_gender_identity in loop_gender_identity:
        currentLoop = loop_gender_identity
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisLoop_gender_identity.rgb)
        if thisLoop_gender_identity != None:
            for paramName in thisLoop_gender_identity:
                globals()[paramName] = thisLoop_gender_identity[paramName]
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'loop_gender_identity'
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "blank_wait" ---
    # create an object to store info about Routine blank_wait
    blank_wait = data.Routine(
        name='blank_wait',
        components=[waittext],
    )
    blank_wait.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for blank_wait
    blank_wait.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    blank_wait.tStart = globalClock.getTime(format='float')
    blank_wait.status = STARTED
    thisExp.addData('blank_wait.started', blank_wait.tStart)
    blank_wait.maxDuration = None
    # keep track of which components have finished
    blank_waitComponents = blank_wait.components
    for thisComponent in blank_wait.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "blank_wait" ---
    blank_wait.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 7.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *waittext* updates
        
        # if waittext is starting this frame...
        if waittext.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            waittext.frameNStart = frameN  # exact frame index
            waittext.tStart = t  # local t and not account for scr refresh
            waittext.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(waittext, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'waittext.started')
            # update status
            waittext.status = STARTED
            waittext.setAutoDraw(True)
        
        # if waittext is active this frame...
        if waittext.status == STARTED:
            # update params
            pass
        
        # if waittext is stopping this frame...
        if waittext.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > waittext.tStartRefresh + 7-frameTolerance:
                # keep track of stop time/frame for later
                waittext.tStop = t  # not accounting for scr refresh
                waittext.tStopRefresh = tThisFlipGlobal  # on global time
                waittext.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'waittext.stopped')
                # update status
                waittext.status = FINISHED
                waittext.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            blank_wait.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in blank_wait.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "blank_wait" ---
    for thisComponent in blank_wait.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for blank_wait
    blank_wait.tStop = globalClock.getTime(format='float')
    blank_wait.tStopRefresh = tThisFlipGlobal
    thisExp.addData('blank_wait.stopped', blank_wait.tStop)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if blank_wait.maxDurationReached:
        routineTimer.addTime(-blank_wait.maxDuration)
    elif blank_wait.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-7.000000)
    thisExp.nextEntry()
    
    # --- Prepare to start Routine "feedback_score_masculine" ---
    # create an object to store info about Routine feedback_score_masculine
    feedback_score_masculine = data.Routine(
        name='feedback_score_masculine',
        components=[image, button_4, text_masculine],
    )
    feedback_score_masculine.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    image.setImage('Feedback_masculine.png')
    # reset button_4 to account for continued clicks & clear times on/off
    button_4.reset()
    # Run 'Begin Routine' code from code_2
    if chooseExperiment:
        continueRoutine = False
    # store start times for feedback_score_masculine
    feedback_score_masculine.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    feedback_score_masculine.tStart = globalClock.getTime(format='float')
    feedback_score_masculine.status = STARTED
    thisExp.addData('feedback_score_masculine.started', feedback_score_masculine.tStart)
    feedback_score_masculine.maxDuration = None
    # keep track of which components have finished
    feedback_score_masculineComponents = feedback_score_masculine.components
    for thisComponent in feedback_score_masculine.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "feedback_score_masculine" ---
    feedback_score_masculine.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *image* updates
        
        # if image is starting this frame...
        if image.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            image.frameNStart = frameN  # exact frame index
            image.tStart = t  # local t and not account for scr refresh
            image.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(image, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'image.started')
            # update status
            image.status = STARTED
            image.setAutoDraw(True)
        
        # if image is active this frame...
        if image.status == STARTED:
            # update params
            pass
        # *button_4* updates
        
        # if button_4 is starting this frame...
        if button_4.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            button_4.frameNStart = frameN  # exact frame index
            button_4.tStart = t  # local t and not account for scr refresh
            button_4.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(button_4, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'button_4.started')
            # update status
            button_4.status = STARTED
            win.callOnFlip(button_4.buttonClock.reset)
            button_4.setAutoDraw(True)
        
        # if button_4 is active this frame...
        if button_4.status == STARTED:
            # update params
            pass
            # check whether button_4 has been pressed
            if button_4.isClicked:
                if not button_4.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    button_4.timesOn.append(button_4.buttonClock.getTime())
                    button_4.timesOff.append(button_4.buttonClock.getTime())
                elif len(button_4.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    button_4.timesOff[-1] = button_4.buttonClock.getTime()
                if not button_4.wasClicked:
                    # end routine when button_4 is clicked
                    continueRoutine = False
                if not button_4.wasClicked:
                    # run callback code when button_4 is clicked
                    pass
        # take note of whether button_4 was clicked, so that next frame we know if clicks are new
        button_4.wasClicked = button_4.isClicked and button_4.status == STARTED
        
        # *text_masculine* updates
        
        # if text_masculine is starting this frame...
        if text_masculine.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_masculine.frameNStart = frameN  # exact frame index
            text_masculine.tStart = t  # local t and not account for scr refresh
            text_masculine.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_masculine, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_masculine.started')
            # update status
            text_masculine.status = STARTED
            text_masculine.setAutoDraw(True)
        
        # if text_masculine is active this frame...
        if text_masculine.status == STARTED:
            # update params
            pass
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            feedback_score_masculine.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in feedback_score_masculine.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "feedback_score_masculine" ---
    for thisComponent in feedback_score_masculine.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for feedback_score_masculine
    feedback_score_masculine.tStop = globalClock.getTime(format='float')
    feedback_score_masculine.tStopRefresh = tThisFlipGlobal
    thisExp.addData('feedback_score_masculine.stopped', feedback_score_masculine.tStop)
    thisExp.addData('button_4.numClicks', button_4.numClicks)
    if button_4.numClicks:
       thisExp.addData('button_4.timesOn', button_4.timesOn)
       thisExp.addData('button_4.timesOff', button_4.timesOff)
    else:
       thisExp.addData('button_4.timesOn', "")
       thisExp.addData('button_4.timesOff', "")
    thisExp.nextEntry()
    # the Routine "feedback_score_masculine" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "feedback_score_feminine" ---
    # create an object to store info about Routine feedback_score_feminine
    feedback_score_feminine = data.Routine(
        name='feedback_score_feminine',
        components=[button_9, text_feminine, image_2],
    )
    feedback_score_feminine.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # reset button_9 to account for continued clicks & clear times on/off
    button_9.reset()
    image_2.setImage('Feedback_feminine.png')
    # Run 'Begin Routine' code from code_3
    if not chooseExperiment:
        continueRoutine = False
    # store start times for feedback_score_feminine
    feedback_score_feminine.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    feedback_score_feminine.tStart = globalClock.getTime(format='float')
    feedback_score_feminine.status = STARTED
    thisExp.addData('feedback_score_feminine.started', feedback_score_feminine.tStart)
    feedback_score_feminine.maxDuration = None
    # keep track of which components have finished
    feedback_score_feminineComponents = feedback_score_feminine.components
    for thisComponent in feedback_score_feminine.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "feedback_score_feminine" ---
    feedback_score_feminine.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        # *button_9* updates
        
        # if button_9 is starting this frame...
        if button_9.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            button_9.frameNStart = frameN  # exact frame index
            button_9.tStart = t  # local t and not account for scr refresh
            button_9.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(button_9, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'button_9.started')
            # update status
            button_9.status = STARTED
            win.callOnFlip(button_9.buttonClock.reset)
            button_9.setAutoDraw(True)
        
        # if button_9 is active this frame...
        if button_9.status == STARTED:
            # update params
            pass
            # check whether button_9 has been pressed
            if button_9.isClicked:
                if not button_9.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    button_9.timesOn.append(button_9.buttonClock.getTime())
                    button_9.timesOff.append(button_9.buttonClock.getTime())
                elif len(button_9.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    button_9.timesOff[-1] = button_9.buttonClock.getTime()
                if not button_9.wasClicked:
                    # end routine when button_9 is clicked
                    continueRoutine = False
                if not button_9.wasClicked:
                    # run callback code when button_9 is clicked
                    pass
        # take note of whether button_9 was clicked, so that next frame we know if clicks are new
        button_9.wasClicked = button_9.isClicked and button_9.status == STARTED
        
        # *text_feminine* updates
        
        # if text_feminine is starting this frame...
        if text_feminine.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_feminine.frameNStart = frameN  # exact frame index
            text_feminine.tStart = t  # local t and not account for scr refresh
            text_feminine.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_feminine, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_feminine.started')
            # update status
            text_feminine.status = STARTED
            text_feminine.setAutoDraw(True)
        
        # if text_feminine is active this frame...
        if text_feminine.status == STARTED:
            # update params
            pass
        
        # *image_2* updates
        
        # if image_2 is starting this frame...
        if image_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            image_2.frameNStart = frameN  # exact frame index
            image_2.tStart = t  # local t and not account for scr refresh
            image_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(image_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'image_2.started')
            # update status
            image_2.status = STARTED
            image_2.setAutoDraw(True)
        
        # if image_2 is active this frame...
        if image_2.status == STARTED:
            # update params
            pass
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            feedback_score_feminine.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in feedback_score_feminine.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "feedback_score_feminine" ---
    for thisComponent in feedback_score_feminine.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for feedback_score_feminine
    feedback_score_feminine.tStop = globalClock.getTime(format='float')
    feedback_score_feminine.tStopRefresh = tThisFlipGlobal
    thisExp.addData('feedback_score_feminine.stopped', feedback_score_feminine.tStop)
    thisExp.addData('button_9.numClicks', button_9.numClicks)
    if button_9.numClicks:
       thisExp.addData('button_9.timesOn', button_9.timesOn)
       thisExp.addData('button_9.timesOff', button_9.timesOff)
    else:
       thisExp.addData('button_9.timesOn', "")
       thisExp.addData('button_9.timesOff', "")
    thisExp.nextEntry()
    # the Routine "feedback_score_feminine" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "pretext_group_based_dom" ---
    # create an object to store info about Routine pretext_group_based_dom
    pretext_group_based_dom = data.Routine(
        name='pretext_group_based_dom',
        components=[pretext_group_based_dominance, button_5],
    )
    pretext_group_based_dom.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # reset button_5 to account for continued clicks & clear times on/off
    button_5.reset()
    # store start times for pretext_group_based_dom
    pretext_group_based_dom.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    pretext_group_based_dom.tStart = globalClock.getTime(format='float')
    pretext_group_based_dom.status = STARTED
    thisExp.addData('pretext_group_based_dom.started', pretext_group_based_dom.tStart)
    pretext_group_based_dom.maxDuration = None
    # keep track of which components have finished
    pretext_group_based_domComponents = pretext_group_based_dom.components
    for thisComponent in pretext_group_based_dom.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "pretext_group_based_dom" ---
    pretext_group_based_dom.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *pretext_group_based_dominance* updates
        
        # if pretext_group_based_dominance is starting this frame...
        if pretext_group_based_dominance.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            pretext_group_based_dominance.frameNStart = frameN  # exact frame index
            pretext_group_based_dominance.tStart = t  # local t and not account for scr refresh
            pretext_group_based_dominance.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(pretext_group_based_dominance, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'pretext_group_based_dominance.started')
            # update status
            pretext_group_based_dominance.status = STARTED
            pretext_group_based_dominance.setAutoDraw(True)
        
        # if pretext_group_based_dominance is active this frame...
        if pretext_group_based_dominance.status == STARTED:
            # update params
            pass
        # *button_5* updates
        
        # if button_5 is starting this frame...
        if button_5.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            button_5.frameNStart = frameN  # exact frame index
            button_5.tStart = t  # local t and not account for scr refresh
            button_5.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(button_5, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'button_5.started')
            # update status
            button_5.status = STARTED
            win.callOnFlip(button_5.buttonClock.reset)
            button_5.setAutoDraw(True)
        
        # if button_5 is active this frame...
        if button_5.status == STARTED:
            # update params
            pass
            # check whether button_5 has been pressed
            if button_5.isClicked:
                if not button_5.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    button_5.timesOn.append(button_5.buttonClock.getTime())
                    button_5.timesOff.append(button_5.buttonClock.getTime())
                elif len(button_5.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    button_5.timesOff[-1] = button_5.buttonClock.getTime()
                if not button_5.wasClicked:
                    # end routine when button_5 is clicked
                    continueRoutine = False
                if not button_5.wasClicked:
                    # run callback code when button_5 is clicked
                    pass
        # take note of whether button_5 was clicked, so that next frame we know if clicks are new
        button_5.wasClicked = button_5.isClicked and button_5.status == STARTED
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            pretext_group_based_dom.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in pretext_group_based_dom.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "pretext_group_based_dom" ---
    for thisComponent in pretext_group_based_dom.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for pretext_group_based_dom
    pretext_group_based_dom.tStop = globalClock.getTime(format='float')
    pretext_group_based_dom.tStopRefresh = tThisFlipGlobal
    thisExp.addData('pretext_group_based_dom.stopped', pretext_group_based_dom.tStop)
    thisExp.addData('button_5.numClicks', button_5.numClicks)
    if button_5.numClicks:
       thisExp.addData('button_5.timesOn', button_5.timesOn)
       thisExp.addData('button_5.timesOff', button_5.timesOff)
    else:
       thisExp.addData('button_5.timesOn', "")
       thisExp.addData('button_5.timesOff', "")
    thisExp.nextEntry()
    # the Routine "pretext_group_based_dom" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    loop_group_based_dominance = data.TrialHandler2(
        name='loop_group_based_dominance',
        nReps=1.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('items_group_based_dominance.xlsx'), 
        seed=None, 
    )
    thisExp.addLoop(loop_group_based_dominance)  # add the loop to the experiment
    thisLoop_group_based_dominance = loop_group_based_dominance.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisLoop_group_based_dominance.rgb)
    if thisLoop_group_based_dominance != None:
        for paramName in thisLoop_group_based_dominance:
            globals()[paramName] = thisLoop_group_based_dominance[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisLoop_group_based_dominance in loop_group_based_dominance:
        currentLoop = loop_group_based_dominance
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisLoop_group_based_dominance.rgb)
        if thisLoop_group_based_dominance != None:
            for paramName in thisLoop_group_based_dominance:
                globals()[paramName] = thisLoop_group_based_dominance[paramName]
        
        # --- Prepare to start Routine "trial_group_based_dominance" ---
        # create an object to store info about Routine trial_group_based_dominance
        trial_group_based_dominance = data.Routine(
            name='trial_group_based_dominance',
            components=[slider_3, text5],
        )
        trial_group_based_dominance.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        slider_3.reset()
        text5.setText(items_group_based_dominance)
        # store start times for trial_group_based_dominance
        trial_group_based_dominance.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        trial_group_based_dominance.tStart = globalClock.getTime(format='float')
        trial_group_based_dominance.status = STARTED
        thisExp.addData('trial_group_based_dominance.started', trial_group_based_dominance.tStart)
        trial_group_based_dominance.maxDuration = None
        # keep track of which components have finished
        trial_group_based_dominanceComponents = trial_group_based_dominance.components
        for thisComponent in trial_group_based_dominance.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "trial_group_based_dominance" ---
        # if trial has changed, end Routine now
        if isinstance(loop_group_based_dominance, data.TrialHandler2) and thisLoop_group_based_dominance.thisN != loop_group_based_dominance.thisTrial.thisN:
            continueRoutine = False
        trial_group_based_dominance.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *slider_3* updates
            
            # if slider_3 is starting this frame...
            if slider_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                slider_3.frameNStart = frameN  # exact frame index
                slider_3.tStart = t  # local t and not account for scr refresh
                slider_3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(slider_3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'slider_3.started')
                # update status
                slider_3.status = STARTED
                slider_3.setAutoDraw(True)
            
            # if slider_3 is active this frame...
            if slider_3.status == STARTED:
                # update params
                pass
            
            # Check slider_3 for response to end Routine
            if slider_3.getRating() is not None and slider_3.status == STARTED:
                continueRoutine = False
            
            # *text5* updates
            
            # if text5 is starting this frame...
            if text5.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text5.frameNStart = frameN  # exact frame index
                text5.tStart = t  # local t and not account for scr refresh
                text5.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text5, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text5.started')
                # update status
                text5.status = STARTED
                text5.setAutoDraw(True)
            
            # if text5 is active this frame...
            if text5.status == STARTED:
                # update params
                pass
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                trial_group_based_dominance.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in trial_group_based_dominance.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "trial_group_based_dominance" ---
        for thisComponent in trial_group_based_dominance.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for trial_group_based_dominance
        trial_group_based_dominance.tStop = globalClock.getTime(format='float')
        trial_group_based_dominance.tStopRefresh = tThisFlipGlobal
        thisExp.addData('trial_group_based_dominance.stopped', trial_group_based_dominance.tStop)
        loop_group_based_dominance.addData('slider_3.response', slider_3.getRating())
        loop_group_based_dominance.addData('slider_3.rt', slider_3.getRT())
        # the Routine "trial_group_based_dominance" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'loop_group_based_dominance'
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "pretext_political_conservatism" ---
    # create an object to store info about Routine pretext_political_conservatism
    pretext_political_conservatism = data.Routine(
        name='pretext_political_conservatism',
        components=[pretext_political_position, button_6],
    )
    pretext_political_conservatism.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # reset button_6 to account for continued clicks & clear times on/off
    button_6.reset()
    # store start times for pretext_political_conservatism
    pretext_political_conservatism.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    pretext_political_conservatism.tStart = globalClock.getTime(format='float')
    pretext_political_conservatism.status = STARTED
    thisExp.addData('pretext_political_conservatism.started', pretext_political_conservatism.tStart)
    pretext_political_conservatism.maxDuration = None
    # keep track of which components have finished
    pretext_political_conservatismComponents = pretext_political_conservatism.components
    for thisComponent in pretext_political_conservatism.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "pretext_political_conservatism" ---
    pretext_political_conservatism.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *pretext_political_position* updates
        
        # if pretext_political_position is starting this frame...
        if pretext_political_position.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            pretext_political_position.frameNStart = frameN  # exact frame index
            pretext_political_position.tStart = t  # local t and not account for scr refresh
            pretext_political_position.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(pretext_political_position, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'pretext_political_position.started')
            # update status
            pretext_political_position.status = STARTED
            pretext_political_position.setAutoDraw(True)
        
        # if pretext_political_position is active this frame...
        if pretext_political_position.status == STARTED:
            # update params
            pass
        # *button_6* updates
        
        # if button_6 is starting this frame...
        if button_6.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            button_6.frameNStart = frameN  # exact frame index
            button_6.tStart = t  # local t and not account for scr refresh
            button_6.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(button_6, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'button_6.started')
            # update status
            button_6.status = STARTED
            win.callOnFlip(button_6.buttonClock.reset)
            button_6.setAutoDraw(True)
        
        # if button_6 is active this frame...
        if button_6.status == STARTED:
            # update params
            pass
            # check whether button_6 has been pressed
            if button_6.isClicked:
                if not button_6.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    button_6.timesOn.append(button_6.buttonClock.getTime())
                    button_6.timesOff.append(button_6.buttonClock.getTime())
                elif len(button_6.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    button_6.timesOff[-1] = button_6.buttonClock.getTime()
                if not button_6.wasClicked:
                    # end routine when button_6 is clicked
                    continueRoutine = False
                if not button_6.wasClicked:
                    # run callback code when button_6 is clicked
                    pass
        # take note of whether button_6 was clicked, so that next frame we know if clicks are new
        button_6.wasClicked = button_6.isClicked and button_6.status == STARTED
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            pretext_political_conservatism.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in pretext_political_conservatism.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "pretext_political_conservatism" ---
    for thisComponent in pretext_political_conservatism.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for pretext_political_conservatism
    pretext_political_conservatism.tStop = globalClock.getTime(format='float')
    pretext_political_conservatism.tStopRefresh = tThisFlipGlobal
    thisExp.addData('pretext_political_conservatism.stopped', pretext_political_conservatism.tStop)
    thisExp.addData('button_6.numClicks', button_6.numClicks)
    if button_6.numClicks:
       thisExp.addData('button_6.timesOn', button_6.timesOn)
       thisExp.addData('button_6.timesOff', button_6.timesOff)
    else:
       thisExp.addData('button_6.timesOn', "")
       thisExp.addData('button_6.timesOff', "")
    thisExp.nextEntry()
    # the Routine "pretext_political_conservatism" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    loop_political_position = data.TrialHandler2(
        name='loop_political_position',
        nReps=1.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('items_political_position.xlsx'), 
        seed=None, 
    )
    thisExp.addLoop(loop_political_position)  # add the loop to the experiment
    thisLoop_political_position = loop_political_position.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisLoop_political_position.rgb)
    if thisLoop_political_position != None:
        for paramName in thisLoop_political_position:
            globals()[paramName] = thisLoop_political_position[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisLoop_political_position in loop_political_position:
        currentLoop = loop_political_position
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisLoop_political_position.rgb)
        if thisLoop_political_position != None:
            for paramName in thisLoop_political_position:
                globals()[paramName] = thisLoop_political_position[paramName]
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'loop_political_position'
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # set up handler to look after randomisation of conditions etc
    loop_system_justification = data.TrialHandler2(
        name='loop_system_justification',
        nReps=1.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('items_system_justification.xlsx'), 
        seed=None, 
    )
    thisExp.addLoop(loop_system_justification)  # add the loop to the experiment
    thisLoop_system_justification = loop_system_justification.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisLoop_system_justification.rgb)
    if thisLoop_system_justification != None:
        for paramName in thisLoop_system_justification:
            globals()[paramName] = thisLoop_system_justification[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisLoop_system_justification in loop_system_justification:
        currentLoop = loop_system_justification
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisLoop_system_justification.rgb)
        if thisLoop_system_justification != None:
            for paramName in thisLoop_system_justification:
                globals()[paramName] = thisLoop_system_justification[paramName]
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'loop_system_justification'
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # set up handler to look after randomisation of conditions etc
    loop_traditionalism = data.TrialHandler2(
        name='loop_traditionalism',
        nReps=1.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('items_traditionalism.xlsx'), 
        seed=None, 
    )
    thisExp.addLoop(loop_traditionalism)  # add the loop to the experiment
    thisLoop_traditionalism = loop_traditionalism.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisLoop_traditionalism.rgb)
    if thisLoop_traditionalism != None:
        for paramName in thisLoop_traditionalism:
            globals()[paramName] = thisLoop_traditionalism[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisLoop_traditionalism in loop_traditionalism:
        currentLoop = loop_traditionalism
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisLoop_traditionalism.rgb)
        if thisLoop_traditionalism != None:
            for paramName in thisLoop_traditionalism:
                globals()[paramName] = thisLoop_traditionalism[paramName]
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'loop_traditionalism'
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "thank_you" ---
    # create an object to store info about Routine thank_you
    thank_you = data.Routine(
        name='thank_you',
        components=[Thank_you, button_8],
    )
    thank_you.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # reset button_8 to account for continued clicks & clear times on/off
    button_8.reset()
    # store start times for thank_you
    thank_you.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    thank_you.tStart = globalClock.getTime(format='float')
    thank_you.status = STARTED
    thisExp.addData('thank_you.started', thank_you.tStart)
    thank_you.maxDuration = None
    # keep track of which components have finished
    thank_youComponents = thank_you.components
    for thisComponent in thank_you.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "thank_you" ---
    thank_you.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *Thank_you* updates
        
        # if Thank_you is starting this frame...
        if Thank_you.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            Thank_you.frameNStart = frameN  # exact frame index
            Thank_you.tStart = t  # local t and not account for scr refresh
            Thank_you.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Thank_you, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'Thank_you.started')
            # update status
            Thank_you.status = STARTED
            Thank_you.setAutoDraw(True)
        
        # if Thank_you is active this frame...
        if Thank_you.status == STARTED:
            # update params
            pass
        # *button_8* updates
        
        # if button_8 is starting this frame...
        if button_8.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            button_8.frameNStart = frameN  # exact frame index
            button_8.tStart = t  # local t and not account for scr refresh
            button_8.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(button_8, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'button_8.started')
            # update status
            button_8.status = STARTED
            win.callOnFlip(button_8.buttonClock.reset)
            button_8.setAutoDraw(True)
        
        # if button_8 is active this frame...
        if button_8.status == STARTED:
            # update params
            pass
            # check whether button_8 has been pressed
            if button_8.isClicked:
                if not button_8.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    button_8.timesOn.append(button_8.buttonClock.getTime())
                    button_8.timesOff.append(button_8.buttonClock.getTime())
                elif len(button_8.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    button_8.timesOff[-1] = button_8.buttonClock.getTime()
                if not button_8.wasClicked:
                    # end routine when button_8 is clicked
                    continueRoutine = False
                if not button_8.wasClicked:
                    # run callback code when button_8 is clicked
                    pass
        # take note of whether button_8 was clicked, so that next frame we know if clicks are new
        button_8.wasClicked = button_8.isClicked and button_8.status == STARTED
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            thank_you.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in thank_you.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "thank_you" ---
    for thisComponent in thank_you.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for thank_you
    thank_you.tStop = globalClock.getTime(format='float')
    thank_you.tStopRefresh = tThisFlipGlobal
    thisExp.addData('thank_you.stopped', thank_you.tStop)
    thisExp.addData('button_8.numClicks', button_8.numClicks)
    if button_8.numClicks:
       thisExp.addData('button_8.timesOn', button_8.timesOn)
       thisExp.addData('button_8.timesOff', button_8.timesOff)
    else:
       thisExp.addData('button_8.timesOn', "")
       thisExp.addData('button_8.timesOff', "")
    thisExp.nextEntry()
    # the Routine "thank_you" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # mark experiment as finished
    endExperiment(thisExp, win=win)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # return console logger level to WARNING
    logging.console.setLevel(logging.WARNING)
    # mark experiment handler as finished
    thisExp.status = FINISHED
    logging.flush()


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win,
        globalClock='float'
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)

/**
 * input.js - keyboard / mouse / pointer-lock aggregation.
 *
 * Systems poll this rather than binding their own listeners, so key handling
 * stays in one place and the HUD can steal focus without fighting the game.
 */

export class Input {
  /**
   * @param {HTMLElement} domElement element that receives pointer lock
   * @param {import('./state.js').EventBus} bus
   */
  constructor(domElement, bus) {
    this.dom = domElement;
    this.bus = bus;

    /** Currently-held physical keys, by KeyboardEvent.code. */
    this.keys = new Set();
    /** Keys that went down this frame - cleared by endFrame(). */
    this.pressed = new Set();
    /** Keys that came up this frame - cleared by endFrame(). */
    this.released = new Set();

    /** Accumulated mouse delta since last frame, in pixels. */
    this.mouseDX = 0;
    this.mouseDY = 0;
    this.wheelDelta = 0;
    this.buttons = new Set();

    this.pointerLocked = false;
    /** Set true while a text field / slider has focus, so WASD does not move the player. */
    this.uiHasFocus = false;

    this._onKeyDown = this._onKeyDown.bind(this);
    this._onKeyUp = this._onKeyUp.bind(this);
    this._onMouseMove = this._onMouseMove.bind(this);
    this._onMouseDown = this._onMouseDown.bind(this);
    this._onMouseUp = this._onMouseUp.bind(this);
    this._onWheel = this._onWheel.bind(this);
    this._onPointerLockChange = this._onPointerLockChange.bind(this);
    this._onBlur = this._onBlur.bind(this);
    this._onContextMenu = (e) => e.preventDefault();

    window.addEventListener('keydown', this._onKeyDown);
    window.addEventListener('keyup', this._onKeyUp);
    window.addEventListener('mousemove', this._onMouseMove);
    window.addEventListener('mousedown', this._onMouseDown);
    window.addEventListener('mouseup', this._onMouseUp);
    window.addEventListener('wheel', this._onWheel, { passive: false });
    window.addEventListener('blur', this._onBlur);
    document.addEventListener('pointerlockchange', this._onPointerLockChange);
    this.dom.addEventListener('contextmenu', this._onContextMenu);
  }

  // -------------------------------------------------------------------------
  // Query API
  // -------------------------------------------------------------------------

  /** Held right now. Returns false while a UI field has focus. */
  isDown(code) {
    return !this.uiHasFocus && this.keys.has(code);
  }
  /** Went down this frame. */
  wasPressed(code) {
    return !this.uiHasFocus && this.pressed.has(code);
  }
  wasReleased(code) {
    return this.released.has(code);
  }
  isMouseDown(button = 0) {
    return this.buttons.has(button);
  }

  /** True if any of the given codes is held - for WASD-or-arrows style bindings. */
  anyDown(...codes) {
    for (const c of codes) if (this.isDown(c)) return true;
    return false;
  }

  requestPointerLock() {
    if (!this.pointerLocked) this.dom.requestPointerLock?.();
  }
  exitPointerLock() {
    if (this.pointerLocked) document.exitPointerLock?.();
  }

  /**
   * Call at the very end of each frame, AFTER all systems have polled.
   * Clears edge-triggered sets and consumed mouse deltas.
   */
  endFrame() {
    this.pressed.clear();
    this.released.clear();
    this.mouseDX = 0;
    this.mouseDY = 0;
    this.wheelDelta = 0;
  }

  dispose() {
    window.removeEventListener('keydown', this._onKeyDown);
    window.removeEventListener('keyup', this._onKeyUp);
    window.removeEventListener('mousemove', this._onMouseMove);
    window.removeEventListener('mousedown', this._onMouseDown);
    window.removeEventListener('mouseup', this._onMouseUp);
    window.removeEventListener('wheel', this._onWheel);
    window.removeEventListener('blur', this._onBlur);
    document.removeEventListener('pointerlockchange', this._onPointerLockChange);
    this.dom.removeEventListener('contextmenu', this._onContextMenu);
  }

  // -------------------------------------------------------------------------
  // Listeners
  // -------------------------------------------------------------------------

  _onKeyDown(e) {
    // Let the browser have its reload / devtools / fullscreen shortcuts.
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    if (!this.keys.has(e.code)) this.pressed.add(e.code);
    this.keys.add(e.code);
    // Space and the arrows scroll the page otherwise, which fights the controller.
    if (['Space', 'ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'Tab'].includes(e.code)) {
      if (!this.uiHasFocus) e.preventDefault();
    }
  }

  _onKeyUp(e) {
    this.keys.delete(e.code);
    this.released.add(e.code);
  }

  _onMouseMove(e) {
    if (!this.pointerLocked) return;
    // movementX/Y occasionally spikes on pointer-lock acquisition in Chrome;
    // clamp so a single bad event cannot whip the camera around.
    const dx = Math.max(-200, Math.min(200, e.movementX || 0));
    const dy = Math.max(-200, Math.min(200, e.movementY || 0));
    this.mouseDX += dx;
    this.mouseDY += dy;
  }

  _onMouseDown(e) {
    this.buttons.add(e.button);
  }
  _onMouseUp(e) {
    this.buttons.delete(e.button);
  }
  _onWheel(e) {
    if (this.uiHasFocus) return;
    e.preventDefault();
    this.wheelDelta += e.deltaY;
  }

  _onPointerLockChange() {
    this.pointerLocked = document.pointerLockElement === this.dom;
    this.bus?.emit('input:pointerlock', this.pointerLocked);
    // Dropping lock while keys are held would leave the player sliding forever.
    if (!this.pointerLocked) this.keys.clear();
  }

  /** Alt-tabbing away must not leave movement keys stuck down. */
  _onBlur() {
    this.keys.clear();
    this.buttons.clear();
    this.pressed.clear();
  }
}

/**
 * 硬件交互守门人
 * 核心逻辑：确保任何参数申请都在安全物理范围内
 */

export interface HardwareCommand {
  ballGap: number; // 球隙间距
  tungstenGap: number; // 钨针间距
  capacitorLevel: number; // 电容级数
}

const HARDWARE_LIMITS = {
  ballGap: { min: 1, max: 20 },
  tungstenGap: { min: 0.5, max: 10 },
  capacitorLevel: { min: 1, max: 5 }
};

export class HardwareGuard {
  static validate(params: HardwareCommand): boolean {
    if (params.ballGap < HARDWARE_LIMITS.ballGap.min || params.ballGap > HARDWARE_LIMITS.ballGap.max) return false;
    if (params.tungstenGap < HARDWARE_LIMITS.tungstenGap.min || params.tungstenGap > HARDWARE_LIMITS.tungstenGap.max) return false;
    if (params.capacitorLevel < HARDWARE_LIMITS.capacitorLevel.min || params.capacitorLevel > HARDWARE_LIMITS.capacitorLevel.max) return false;
    return true;
  }

  static async sendToMCU(command: HardwareCommand) {
    if (!this.validate(command)) {
      throw new Error("Safety Violation: Parameters out of physical hardware bounds.");
    }
    console.log("Sending command to MCU:", command);
    // 模拟真实的硬件通信
    return { success: true, timestamp: Date.now() };
  }
}

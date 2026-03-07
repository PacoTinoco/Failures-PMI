import { NextRequest, NextResponse } from 'next/server'
import { getServerSession } from 'next-auth'
import { authOptions } from '@/lib/auth'
import { prisma } from '@/lib/prisma'
import { parseExcelFile } from '@/lib/excel-parser'

export async function POST(request: NextRequest) {
  try {
    const session = await getServerSession(authOptions)

    if (!session) {
      return NextResponse.json(
        { error: 'Unauthorized' },
        { status: 401 }
      )
    }

    const formData = await request.formData()
    const file = formData.get('file') as File

    if (!file) {
      return NextResponse.json(
        { error: 'File is required' },
        { status: 400 }
      )
    }

    const buffer = Buffer.from(await file.arrayBuffer())
    const parsedData = parseExcelFile(buffer)

    if (parsedData.length === 0) {
      return NextResponse.json(
        { success: false, rowsProcessed: 0, errors: ['No valid data found in Excel file'] },
        { status: 400 }
      )
    }

    const errors: string[] = []
    let rowsProcessed = 0

    // Process each row
    for (const row of parsedData) {
      try {
        // Find operator by name
        const operator = await prisma.operator.findFirst({
          where: {
            name: row.operatorName
          }
        })

        if (!operator) {
          errors.push(`Operator not found: ${row.operatorName}`)
          continue
        }

        // Parse week start date
        const weekStartDate = new Date(row.weekStart)
        if (isNaN(weekStartDate.getTime())) {
          errors.push(`Invalid date format for operator ${row.operatorName}: ${row.weekStart}`)
          continue
        }

        const year = weekStartDate.getFullYear()
        const weekNumber = Math.ceil(
          (weekStartDate.getTime() - new Date(year, 0, 1).getTime()) / (7 * 24 * 60 * 60 * 1000)
        )

        // Validate KPI codes
        const kpiCodes = Object.keys(row.kpiValues)
        const validKpis = await prisma.kpiDefinition.findMany({
          where: {
            code: {
              in: kpiCodes
            }
          }
        })

        const validKpiCodes = validKpis.map((k) => k.code)
        const invalidCodes = kpiCodes.filter((code) => !validKpiCodes.includes(code))

        if (invalidCodes.length > 0) {
          errors.push(
            `Invalid KPI codes for ${row.operatorName}: ${invalidCodes.join(', ')}`
          )
          continue
        }

        // Check if record exists
        const existingRecord = await prisma.weeklyRecord.findUnique({
          where: {
            operatorId_weekStart: {
              operatorId: operator.id,
              weekStart: weekStartDate
            }
          }
        })

        if (existingRecord) {
          // Delete existing KPI values
          await prisma.kpiValue.deleteMany({
            where: {
              weeklyRecordId: existingRecord.id
            }
          })

          // Update record
          await prisma.weeklyRecord.update({
            where: { id: existingRecord.id },
            data: {
              year,
              weekNumber,
              createdBy: session.user.id,
              kpiValues: {
                create: Object.entries(row.kpiValues).map(([code, value]) => ({
                  kpiCode: code,
                  value: value as number,
                  objective: row.objectives[code]
                }))
              }
            }
          })
        } else {
          // Create new record
          await prisma.weeklyRecord.create({
            data: {
              operatorId: operator.id,
              weekStart: weekStartDate,
              year,
              weekNumber,
              createdBy: session.user.id,
              kpiValues: {
                create: Object.entries(row.kpiValues).map(([code, value]) => ({
                  kpiCode: code,
                  value: value as number,
                  objective: row.objectives[code]
                }))
              }
            }
          })
        }

        rowsProcessed++
      } catch (error) {
        errors.push(
          `Error processing row for ${row.operatorName}: ${(error as Error).message}`
        )
      }
    }

    // Log the upload
    const machineId = formData.get('machineId') as string
    await prisma.uploadLog.create({
      data: {
        userId: session.user.id,
        filename: file.name,
        machineId: machineId || null,
        rowsProcessed
      }
    })

    return NextResponse.json({
      success: errors.length === 0,
      rowsProcessed,
      errors
    })
  } catch (error) {
    console.error('Error uploading file:', error)
    return NextResponse.json(
      { error: 'Failed to process file upload', success: false, rowsProcessed: 0, errors: [] },
      { status: 500 }
    )
  }
}

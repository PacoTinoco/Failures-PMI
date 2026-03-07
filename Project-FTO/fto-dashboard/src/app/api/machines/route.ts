import { NextRequest, NextResponse } from 'next/server'
import { getServerSession } from 'next-auth'
import { authOptions } from '@/lib/auth'
import { prisma } from '@/lib/prisma'

export async function GET() {
  try {
    const machines = await prisma.machine.findMany({
      include: {
        coordinators: {
          include: {
            operators: true
          }
        }
      }
    })

    const machinesWithStats = machines.map((machine) => ({
      ...machine,
      coordinatorCount: machine.coordinators.length,
      operatorCount: machine.coordinators.reduce(
        (sum, coord) => sum + coord.operators.length,
        0
      )
    }))

    return NextResponse.json({
      machines: machinesWithStats
    })
  } catch (error) {
    console.error('Error fetching machines:', error)
    return NextResponse.json(
      { error: 'Failed to fetch machines' },
      { status: 500 }
    )
  }
}

export async function POST(request: NextRequest) {
  try {
    const session = await getServerSession(authOptions)

    if (!session || session.user.role !== 'ADMIN') {
      return NextResponse.json(
        { error: 'Unauthorized. Admin access required.' },
        { status: 403 }
      )
    }

    const body = await request.json()
    const { name, code } = body

    if (!name || !code) {
      return NextResponse.json(
        { error: 'Name and code are required' },
        { status: 400 }
      )
    }

    const machine = await prisma.machine.create({
      data: {
        name,
        code
      }
    })

    return NextResponse.json(machine, { status: 201 })
  } catch (error) {
    console.error('Error creating machine:', error)
    if ((error as any)?.code === 'P2002') {
      return NextResponse.json(
        { error: 'Machine name or code already exists' },
        { status: 409 }
      )
    }
    return NextResponse.json(
      { error: 'Failed to create machine' },
      { status: 500 }
    )
  }
}
